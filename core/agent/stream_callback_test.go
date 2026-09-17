package agent

import (
	"testing"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
)

func TestStreamCallbackForJobCombinesAgentAndRequestCallbacks(t *testing.T) {
	var agentEvents, firstRequestEvents, secondRequestEvents []cogito.StreamEvent
	a := &Agent{options: &options{
		streamCallback: func(event cogito.StreamEvent) {
			agentEvents = append(agentEvents, event)
		},
	}}
	first := types.NewJob(types.WithStreamCallback(func(event cogito.StreamEvent) {
		firstRequestEvents = append(firstRequestEvents, event)
	}))
	second := types.NewJob(types.WithStreamCallback(func(event cogito.StreamEvent) {
		secondRequestEvents = append(secondRequestEvents, event)
	}))

	firstEvent := cogito.StreamEvent{Content: "first"}
	secondEvent := cogito.StreamEvent{Content: "second"}
	a.streamCallbackForJob(first)(firstEvent)
	a.streamCallbackForJob(second)(secondEvent)

	if len(agentEvents) != 2 || agentEvents[0].Content != "first" || agentEvents[1].Content != "second" {
		t.Fatalf("agent callback events = %#v, want first and second events", agentEvents)
	}
	if len(firstRequestEvents) != 1 || firstRequestEvents[0].Content != "first" {
		t.Fatalf("first request callback events = %#v, want only first event", firstRequestEvents)
	}
	if len(secondRequestEvents) != 1 || secondRequestEvents[0].Content != "second" {
		t.Fatalf("second request callback events = %#v, want only second event", secondRequestEvents)
	}
}

func TestStreamCallbackForJobNilRequestCallbackPreservesAgentCallback(t *testing.T) {
	var events []cogito.StreamEvent
	a := &Agent{options: &options{
		streamCallback: func(event cogito.StreamEvent) {
			events = append(events, event)
		},
	}}

	callback := a.streamCallbackForJob(types.NewJob(types.WithStreamCallback(nil)))
	callback(cogito.StreamEvent{Content: "agent"})

	if len(events) != 1 || events[0].Content != "agent" {
		t.Fatalf("agent callback events = %#v, want agent event", events)
	}
}

func TestStreamCallbackForJobPreservesJobIdentity(t *testing.T) {
	var ids []string
	var global, request int
	a := &Agent{options: &options{
		streamCallback: func(cogito.StreamEvent) { global++ },
		jobStreamCallback: func(job *types.Job, event cogito.StreamEvent) {
			ids = append(ids, job.Metadata[types.MetadataKeyConversationID].(string))
		},
	}}
	first := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "first"}), types.WithStreamCallback(func(cogito.StreamEvent) { request++ }))
	second := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "second"}))
	cb1, cb2 := a.streamCallbackForJob(first), a.streamCallbackForJob(second)
	cb2(cogito.StreamEvent{})
	cb1(cogito.StreamEvent{})
	if len(ids) != 2 || ids[0] != "second" || ids[1] != "first" || global != 2 || request != 1 {
		t.Fatalf("ids=%v global=%d request=%d", ids, global, request)
	}
}
