// Package chat provides the conversation and event lifecycle shared by chat APIs.
package chat

import (
	"encoding/json"
	"time"

	"github.com/mudler/LocalAGI/core/sse"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
)

// Agent is the agent interface needed by the chat lifecycle.
type Agent interface {
	Ask(...types.JobOption) *types.JobResult
	SharedState() *types.AgentSharedState
}

// Run processes one chat turn synchronously. HTTP callers should run it in a
// goroutine after validating the request. An empty conversationID preserves the
// stateless chat behavior. History expires according to last_message_duration.
// send receives SSE envelopes, including conversation and message correlation.
func Run(agent Agent, message, conversationID, messageID string, send func(sse.Envelope)) {
	emit := func(event string, data map[string]any) { emitEvent(send, event, conversationID, messageID, data) }
	// Retry admission here as well as at the HTTP boundary: the existing job may
	// park between request validation and this asynchronous lifecycle starting.
	if live, ok := agent.(interface {
		InjectChat(string, string, string) (bool, error)
	}); ok {
		if handled, err := live.InjectChat(conversationID, message, messageID); handled {
			if err != nil {
				emit("json_error", map[string]any{"error": err.Error()})
				emit("json_message_status", map[string]any{"status": "completed"})
			}
			return
		}
	}

	emit("json_message", map[string]any{"id": messageID + "-user", "sender": "user", "content": message})
	emit("json_message_status", map[string]any{"status": "processing"})
	opts := []types.JobOption{types.WithUUID(messageID), types.WithEventCallback(func(event string, payload any) {
		data, _ := json.Marshal(payload)
		send(sse.NewMessage(string(data)).WithEvent(event))
	})}
	if conversationID != "" {
		opts = append(opts, types.WithConversationHistory(agent.SharedState().ConversationTracker.GetConversation(conversationID)), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: conversationID}))
	}
	opts = append(opts, types.WithText(message))
	response := agent.Ask(opts...)
	switch {
	case response == nil:
		emit("json_error", map[string]any{"error": "agent request failed or was cancelled"})
	case response.Error != nil:
		emit("json_error", map[string]any{"error": response.Error.Error()})
	default:
		if conversationID != "" && !response.ConversationSaved {
			agent.SharedState().ConversationTracker.SetConversation(conversationID, response.Conversation)
		}
		emit("json_message", map[string]any{"id": messageID + "-agent", "sender": "agent", "content": response.Response})
	}
	emit("json_message_status", map[string]any{"status": "completed"})
}

// Stream forwards supported cogito events with the emitting job's identity.
// It is suitable for agent.WithJobStreamCallback; it does not change the
// agent-wide or request-specific streaming callbacks.
func Stream(job *types.Job, event cogito.StreamEvent, send func(sse.Envelope)) {
	data := map[string]any{"type": event.Type}
	switch event.Type {
	case cogito.StreamEventReasoning, cogito.StreamEventContent:
		data["content"] = event.Content
	case cogito.StreamEventToolCall:
		data["tool_name"] = event.ToolName
		data["tool_args"] = event.ToolArgs
	case cogito.StreamEventToolResult:
		data["tool_name"] = event.ToolName
		data["tool_result"] = event.ToolResult
		if event.AgentID != "" {
			data["agent_id"] = event.AgentID
		}
	case cogito.StreamEventDone:
	default:
		return
	}
	conversationID, _ := job.Metadata[types.MetadataKeyConversationID].(string)
	emitEvent(send, "stream_event", conversationID, job.UUID, data)
}

func emitEvent(send func(sse.Envelope), event, conversationID, messageID string, data map[string]any) {
	if conversationID != "" {
		data["conversation_id"] = conversationID
	}
	data["message_id"] = messageID
	data["timestamp"] = time.Now().Format(time.RFC3339)
	// All payload values are strings or cogito's string-based event type.
	encoded, _ := json.Marshal(data)
	send(sse.NewMessage(string(encoded)).WithEvent(event))
}
