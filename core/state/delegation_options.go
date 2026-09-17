package state

import (
	"net/http"

	"github.com/mudler/LocalAGI/core/types"
)

// SubAgentResolver authorizes a pool candidate for a parent job and returns its
// model-visible name. Parent and candidate are actual pool keys. Returning false
// or an empty name hides the candidate. The resolver runs outside the pool lock
// and must support concurrent calls; authorization is rechecked before dispatch.
// Aliases must be unique within each parent's authorized scope.
type SubAgentResolver func(parent string, job *types.Job, candidate string) (name string, allowed bool)

// SetSubAgentResolver installs the host's scope and alias policy. A nil resolver
// restores standalone identity naming. SubAgents allowlists use visible names.
func (a *AgentPool) SetSubAgentResolver(resolver SubAgentResolver) {
	a.Lock()
	defer a.Unlock()
	a.subAgentResolver = resolver
}

// HTTPDoer sends remote-agent requests using the host's transport policy.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// SetRemoteAgentHTTPClient installs the client used for remote delegation,
// unchanged, including its redirect, dial, and timeout policies. A nil client
// restores a pool-owned HTTP client with no fixed timeout. Requests always carry
// the delegation context. The client must support concurrent calls.
func (a *AgentPool) SetRemoteAgentHTTPClient(client HTTPDoer) {
	a.Lock()
	defer a.Unlock()
	a.remoteAgentHTTPClient = client
}

func (a *AgentPool) remoteHTTPClient() HTTPDoer {
	a.Lock()
	defer a.Unlock()
	if a.remoteAgentHTTPClient == nil {
		a.remoteAgentHTTPClient = &http.Client{}
	}
	return a.remoteAgentHTTPClient
}

// resolvedSubAgents maps unique visible aliases to actual keys. Reserve both
// the parent's pool key and its alias so an alias cannot delegate to self.
func resolvedSubAgents(parent string, job *types.Job, configs AgentPoolData, resolver SubAgentResolver) map[string]string {
	resolve := resolver
	if resolve == nil {
		resolve = func(_ string, _ *types.Job, candidate string) (string, bool) { return candidate, true }
	}
	blocked := map[string]bool{parent: true}
	if alias, ok := resolve(parent, job, parent); ok && alias != "" {
		blocked[alias] = true
	}
	resolved := make(map[string]string)
	for candidate := range configs {
		if candidate == parent {
			continue
		}
		alias, ok := resolve(parent, job, candidate)
		if !ok || alias == "" || blocked[alias] {
			continue
		}
		if _, exists := resolved[alias]; exists {
			delete(resolved, alias)
			blocked[alias] = true
			continue
		}
		resolved[alias] = candidate
	}
	return resolved
}
