package state

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

type remoteResponsesRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type remoteResponsesResponse struct {
	Status string `json:"status"`
	Error  *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Output []struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
}

// subAgentProvider snapshots the available delegation targets when a parent
// job starts. Pool agents loaded after the parent agent itself was created are
// therefore available without recreating the parent.
func (a *AgentPool) subAgentProvider(name string, config *AgentConfig) func(job *types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher) {
	enabled := config != nil && config.EnableSubAgents
	var allowedNames []string
	var remoteConfigs []RemoteAgent
	if config != nil {
		allowedNames = append([]string(nil), config.SubAgents...)
		remoteConfigs = append([]RemoteAgent(nil), config.RemoteAgents...)
	}
	return func(job *types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher) {
		if !enabled {
			return nil, nil
		}

		a.Lock()
		poolConfigs := make(AgentPoolData, len(a.pool))
		for peerName, peerConfig := range a.pool {
			poolConfigs[peerName] = peerConfig
		}
		poolAgents := make(map[string]poolAgent, len(a.agents))
		for peerName, peer := range a.agents {
			poolAgents[peerName] = peer
		}
		resolver := a.subAgentResolver
		a.Unlock()

		allowed := make(map[string]struct{}, len(allowedNames))
		for _, peerName := range allowedNames {
			if peerName != "" && peerName != name {
				allowed[peerName] = struct{}{}
			}
		}
		allowAllPeers := len(allowedNames) == 0

		resolved := resolvedSubAgents(name, job, poolConfigs, resolver)
		definitions := make(map[string]cogito.AgentDefinition)
		for visibleName, peerName := range resolved {
			if !allowAllPeers {
				if _, ok := allowed[visibleName]; !ok {
					continue
				}
			}
			peerConfig := poolConfigs[peerName]
			definitions[visibleName] = cogito.AgentDefinition{
				Name: visibleName, Description: peerConfig.Description,
				SystemPrompt: peerConfig.SystemPrompt, Model: peerConfig.Model,
			}
		}

		remotes := make(map[string]RemoteAgent, len(remoteConfigs))
		for _, remote := range remoteConfigs {
			if remote.Name == "" {
				continue
			}
			remotes[remote.Name] = remote
			definitions[remote.Name] = cogito.AgentDefinition{
				Name:        remote.Name,
				Description: remote.Description,
			}
		}

		definitionNames := make([]string, 0, len(definitions))
		for definitionName := range definitions {
			definitionNames = append(definitionNames, definitionName)
		}
		sort.Strings(definitionNames)
		defs := make([]cogito.AgentDefinition, 0, len(definitionNames))
		for _, definitionName := range definitionNames {
			defs = append(defs, definitions[definitionName])
		}

		dispatch := func(ctx context.Context, spec cogito.AgentRunSpec) (cogito.Fragment, error) {
			if remote, ok := remotes[spec.Type]; ok {
				return dispatchRemoteAgent(ctx, a.remoteHTTPClient(), remote, spec)
			}

			a.Lock()
			currentResolver := a.subAgentResolver
			currentConfigs := make(AgentPoolData, len(a.pool))
			for key, cfg := range a.pool {
				currentConfigs[key] = cfg
			}
			a.Unlock()
			_, explicitlyAllowed := allowed[spec.Type]
			peerName, configured := resolved[spec.Type]
			if !configured || (!allowAllPeers && !explicitlyAllowed) {
				if resolver != nil || currentResolver != nil {
					return cogito.Fragment{}, fmt.Errorf("local sub-agent %q is not authorized", spec.Type)
				}
				if explicitlyAllowed {
					return cogito.Fragment{}, fmt.Errorf("allowed local sub-agent %q is not configured", spec.Type)
				}
				return cogito.Fragment{}, cogito.ErrDispatchFallback
			}
			currentResolved := resolvedSubAgents(name, job, currentConfigs, currentResolver)
			if currentPeer, ok := currentResolved[spec.Type]; !ok || currentPeer != peerName {
				return cogito.Fragment{}, fmt.Errorf("local sub-agent %q is no longer authorized", spec.Type)
			}

			peer := poolAgents[peerName]
			if peer == nil {
				return cogito.Fragment{}, fmt.Errorf("local sub-agent %q is not running", spec.Type)
			}

			parentID := job.UUID
			if conversationID, ok := job.Metadata[types.MetadataKeyConversationID].(string); ok && conversationID != "" {
				parentID = conversationID
			}
			metadata := make(map[string]any, len(job.Metadata)+4)
			for key, value := range job.Metadata {
				metadata[key] = value
			}
			metadata[types.MetadataKeyConversationID] = parentID
			metadata["parent_agent_id"] = name
			if rootMessage, ok := metadata[types.MetadataKeyParentMessageID].(string); !ok || rootMessage == "" {
				metadata[types.MetadataKeyParentMessageID] = job.UUID
			}
			metadata[types.MetadataKeyDelegationID] = spec.ID
			result := peer.Ask(
				types.WithText(spec.Task),
				types.WithContext(ctx),
				types.WithInteractionHandler(job.InteractionHandler),
				types.WithEventCallback(job.EventCallback),
				types.WithMetadata(metadata),
			)
			if result == nil {
				return cogito.Fragment{}, fmt.Errorf("local sub-agent %q returned no result", spec.Type)
			}
			if result.Error != nil {
				return cogito.Fragment{}, fmt.Errorf("local sub-agent %q failed: %w", spec.Type, result.Error)
			}
			return cogito.NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: result.Response}), nil
		}

		return defs, dispatch
	}
}

// poolAgent is the portion of agent.Agent used by delegation. The concrete
// pool map is copied into this interface map to keep dispatch outside the pool
// lock while preserving a consistent per-job snapshot.
type poolAgent interface {
	Ask(opts ...types.JobOption) *types.JobResult
}

func dispatchRemoteAgent(ctx context.Context, client HTTPDoer, remote RemoteAgent, spec cogito.AgentRunSpec) (cogito.Fragment, error) {
	displayName := redactRemoteSecret(remote.Name, remote.APIKey)
	payload, err := json.Marshal(remoteResponsesRequest{Model: remote.Name, Input: spec.Task})
	if err != nil {
		return cogito.Fragment{}, fmt.Errorf("remote agent %q request encoding failed", displayName)
	}

	endpoint := strings.TrimRight(remote.URL, "/") + "/v1/responses"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return cogito.Fragment{}, fmt.Errorf("remote agent %q request creation failed", displayName)
	}
	req.Header.Set("Content-Type", "application/json")
	if remote.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+remote.APIKey)
	}

	response, err := client.Do(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return cogito.Fragment{}, ctxErr
		}
		return cogito.Fragment{}, fmt.Errorf("remote agent %q request failed", displayName)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return cogito.Fragment{}, fmt.Errorf("remote agent %q returned HTTP status %d", displayName, response.StatusCode)
	}

	var decoded remoteResponsesResponse
	decoder := json.NewDecoder(io.LimitReader(response.Body, 16<<20))
	if err := decoder.Decode(&decoded); err != nil {
		return cogito.Fragment{}, fmt.Errorf("remote agent %q returned an invalid response", displayName)
	}
	if decoded.Error != nil || decoded.Status == "failed" {
		code, message := "", ""
		if decoded.Error != nil {
			code = redactRemoteSecret(decoded.Error.Code, remote.APIKey)
			message = redactRemoteSecret(decoded.Error.Message, remote.APIKey)
		}
		if code == "" && message == "" {
			message = "response status is failed"
		}
		return cogito.Fragment{}, fmt.Errorf("remote agent %q failed (%s): %s", displayName, code, message)
	}

	var text strings.Builder
	for _, output := range decoded.Output {
		for _, content := range output.Content {
			if content.Type == "output_text" {
				text.WriteString(content.Text)
			}
		}
	}
	return cogito.NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: text.String()}), nil
}

func redactRemoteSecret(value, secret string) string {
	if secret == "" {
		return value
	}
	return strings.ReplaceAll(value, secret, "[REDACTED]")
}
