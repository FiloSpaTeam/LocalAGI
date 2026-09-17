package chat

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/mudler/LocalAGI/core/sse"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

type testAgent struct {
	cancelled bool
	state     *types.AgentSharedState
	jobs      []*types.Job
	result    *types.JobResult
}

func (a *testAgent) SharedState() *types.AgentSharedState { return a.state }
func (a *testAgent) Ask(opts ...types.JobOption) *types.JobResult {
	job := types.NewJob(opts...)
	a.jobs = append(a.jobs, job)
	if a.cancelled {
		return nil
	}
	if a.result != nil {
		return a.result
	}
	return &types.JobResult{Response: "reply", Conversation: append(job.ConversationHistory, openai.ChatCompletionMessage{Role: "assistant", Content: "reply"})}
}
func discard(sse.Envelope) {}

func TestConversationRoundTrip(t *testing.T) {
	a := &testAgent{state: types.NewAgentSharedState(time.Hour)}
	Run(a, "first", "opaque / id", "m1", discard)
	Run(a, "second", "opaque / id", "m2", discard)
	want := []openai.ChatCompletionMessage{{Role: "user", Content: "first"}, {Role: "assistant", Content: "reply"}, {Role: "user", Content: "second"}}
	if !reflect.DeepEqual(a.jobs[1].ConversationHistory, want) {
		t.Fatalf("history = %#v", a.jobs[1].ConversationHistory)
	}
	if a.jobs[1].Metadata[types.MetadataKeyConversationID] != "opaque / id" {
		t.Fatal("missing conversation metadata")
	}
	if a.jobs[1].UUID != "m2" {
		t.Fatal("missing message identity")
	}
	if len(a.state.ConversationTracker.GetConversation("opaque / id")) != 4 {
		t.Fatal("completed conversation not saved")
	}
	Run(a, "other", "other", "m3", discard)
	if len(a.jobs[2].ConversationHistory) != 1 {
		t.Fatal("conversation leaked")
	}
}

func TestAnonymousChatIsStateless(t *testing.T) {
	a := &testAgent{state: types.NewAgentSharedState(time.Hour)}
	Run(a, "first", "", "m1", discard)
	Run(a, "second", "", "m2", discard)
	if len(a.jobs[1].ConversationHistory) != 1 || len(a.state.ConversationTracker.GetConversation("")) != 0 {
		t.Fatal("anonymous chat retained history")
	}
	if _, ok := a.jobs[1].Metadata[types.MetadataKeyConversationID]; ok {
		t.Fatal("anonymous job has conversation identity")
	}
}

func TestExpiredConversationStartsFresh(t *testing.T) {
	a := &testAgent{state: types.NewAgentSharedState(-time.Second)}
	a.state.ConversationTracker.SetConversation("expired", []openai.ChatCompletionMessage{{Role: "user", Content: "old"}})
	Run(a, "new", "expired", "m", discard)
	if len(a.jobs[0].ConversationHistory) != 1 || a.jobs[0].ConversationHistory[0].Content != "new" {
		t.Fatal("expired history reused")
	}
}

func TestChatEventsScopedAndLegacyCompatible(t *testing.T) {
	for _, id := range []string{"", "conversation"} {
		for _, failed := range []bool{false, true} {
			a := &testAgent{state: types.NewAgentSharedState(time.Hour)}
			if failed {
				a.result = &types.JobResult{Error: errors.New("failed")}
			}
			var events []string
			Run(a, "hello", id, "message", func(e sse.Envelope) {
				m := e.(*sse.Message)
				events = append(events, m.Event)
				var data map[string]any
				if err := json.Unmarshal([]byte(m.Data), &data); err != nil {
					t.Fatal(err)
				}
				if id != "" && data["conversation_id"] != id {
					t.Fatalf("unscoped %s: %s", m.Event, m.Data)
				}
				if id == "" {
					if _, ok := data["conversation_id"]; ok {
						t.Fatal("legacy payload includes conversation id")
					}
				}
				if data["message_id"] != "message" {
					t.Fatalf("missing message correlation: %s", m.Data)
				}
			})
			reply := "json_message"
			if failed {
				reply = "json_error"
			}
			if !reflect.DeepEqual(events, []string{"json_message", "json_message_status", reply, "json_message_status"}) {
				t.Fatalf("events = %v", events)
			}
		}
	}
}

func TestStreamEventsCarryJobIdentity(t *testing.T) {
	for _, kind := range []cogito.StreamEventType{cogito.StreamEventReasoning, cogito.StreamEventContent, cogito.StreamEventToolCall, cogito.StreamEventDone} {
		job := types.NewJob(types.WithUUID("message"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
		var got *sse.Message
		Stream(job, cogito.StreamEvent{Type: kind, Content: "text", ToolName: "tool", ToolArgs: "{}"}, func(e sse.Envelope) { got = e.(*sse.Message) })
		if got == nil {
			t.Fatal("missing stream event")
		}
		var data map[string]any
		if err := json.Unmarshal([]byte(got.Data), &data); err != nil {
			t.Fatal(err)
		}
		if data["conversation_id"] != "conversation" || data["message_id"] != "message" {
			t.Fatalf("unscoped event: %s", got.Data)
		}
	}
}

func TestFailedChatPreservesSavedHistory(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		a := &testAgent{state: types.NewAgentSharedState(time.Hour), cancelled: cancelled, result: &types.JobResult{Error: errors.New("failed")}}
		original := []openai.ChatCompletionMessage{{Role: "user", Content: "saved"}}
		a.state.ConversationTracker.SetConversation("id", original)
		var failure, completed bool
		Run(a, "new", "id", "message", func(e sse.Envelope) {
			m := e.(*sse.Message)
			var data map[string]any
			if err := json.Unmarshal([]byte(m.Data), &data); err != nil {
				t.Fatal(err)
			}
			if data["conversation_id"] != "id" {
				t.Fatal("unscoped failure event")
			}
			if m.Event == "json_error" {
				failure = true
			}
			if data["status"] == "completed" {
				completed = true
			}
		})
		if !failure || !completed {
			t.Fatal("missing error or terminal status")
		}
		if !reflect.DeepEqual(a.state.ConversationTracker.GetConversation("id"), original) {
			t.Fatal("failed turn replaced history")
		}
	}
}

func TestLegacyStreamPayload(t *testing.T) {
	job := types.NewJob(types.WithUUID("message"))
	var events []*sse.Message
	send := func(e sse.Envelope) { events = append(events, e.(*sse.Message)) }
	Stream(job, cogito.StreamEvent{Type: cogito.StreamEventContent, Content: "text"}, send)
	Stream(job, cogito.StreamEvent{Type: cogito.StreamEventToolCall, ToolName: "tool", ToolArgs: "{}"}, send)
	Stream(job, cogito.StreamEvent{Type: "unknown"}, send)
	if len(events) != 2 {
		t.Fatalf("events = %d", len(events))
	}
	for i, event := range events {
		var data map[string]any
		if err := json.Unmarshal([]byte(event.Data), &data); err != nil {
			t.Fatal(err)
		}
		if _, exists := data["conversation_id"]; exists {
			t.Fatal("anonymous stream scoped")
		}
		if i == 0 && data["content"] != "text" {
			t.Fatal("content lost")
		}
		if i == 1 && (data["tool_name"] != "tool" || data["tool_args"] != "{}") {
			t.Fatal("tool call lost")
		}
	}
}
