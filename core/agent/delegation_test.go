package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

func TestBackgroundDelegationParksAndAcceptsChat(t *testing.T) {
	release := make(chan struct{})
	childStarted := make(chan struct{})
	llm := &interactiveLLM{turns: []interactiveTurn{{"spawn_agent", `{"agent_type":"coder","task":"implement","background":true}`}}}
	a, events := interactiveAgent(t, llm, WithSubAgents(true), WithMaxEvaluationLoops(10), WithSubAgentProvider(func(job *types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher) {
		return []cogito.AgentDefinition{{Name: "coder", Description: "coding"}}, func(ctx context.Context, spec cogito.AgentRunSpec) (cogito.Fragment, error) {
			close(childStarted)
			select {
			case <-release:
				return cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "implemented"), nil
			case <-ctx.Done():
				return cogito.NewEmptyFragment(), ctx.Err()
			}
		}
	}))
	job := types.NewJob(types.WithText("start"), types.WithUUID("root"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	done := make(chan *types.JobResult, 1)
	go func() { a.consumeJob(job, UserRole); done <- job.Result }()
	awaitInteraction(t, events, "sub_agent")
	first := awaitInteraction(t, events, "json_message")
	if first["content"] != "done" || first["conversation_id"] != "conversation" {
		t.Fatalf("park reply = %v", first)
	}
	history := a.SharedState().ConversationTracker.GetConversation("conversation")
	if len(history) < 4 {
		t.Fatalf("parked history lost tool messages: %+v", history)
	}
	if ok, err := a.InjectChat("other", "wrong", "message-other"); ok || err != nil {
		t.Fatal("cross conversation injection")
	}
	if ok, err := a.InjectChat("conversation", "follow up", "message-two"); !ok || err != nil {
		t.Fatalf("injection=%v,%v", ok, err)
	}
	// Skip the injected user echo; the following assistant reply is the second park.
	var second map[string]any
	for {
		second = awaitInteraction(t, events, "json_message")
		if second["sender"] == "agent" {
			break
		}
	}
	if second["id"] == first["id"] {
		t.Fatal("park replies reused an id")
	}
	if job.GetContext().Err() != nil {
		t.Fatal("injection cancelled the live job")
	}
	close(release)
	result := awaitInteractiveResult(t, done)
	if result.Error != nil {
		t.Fatal(result.Error)
	}
	var follow, completion bool
	for _, m := range result.Conversation {
		follow = follow || m.Content == "follow up"
		completion = completion || strings.Contains(m.Content, "implemented")
	}
	if !follow || !completion {
		t.Fatalf("missing injected content: %+v", result.Conversation)
	}
	if ok, err := a.InjectChat("conversation", "new turn", "message-three"); ok || err != nil {
		t.Fatal("completed loop still accepts messages")
	}
	if len(a.SharedState().ConversationTracker.GetConversation("conversation")) != len(result.Conversation) {
		t.Fatal("final conversation not saved before releasing live loop")
	}
}

func TestBackgroundDelegationCancelledOnPause(t *testing.T) {
	cancelled := make(chan struct{})
	llm := &interactiveLLM{turns: []interactiveTurn{{"spawn_agent", `{"agent_type":"coder","task":"implement","background":true}`}}}
	a, events := interactiveAgent(t, llm, WithSubAgents(true), WithSubAgentProvider(func(*types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher) {
		return []cogito.AgentDefinition{{Name: "coder"}}, func(ctx context.Context, _ cogito.AgentRunSpec) (cogito.Fragment, error) {
			<-ctx.Done()
			close(cancelled)
			return cogito.NewEmptyFragment(), ctx.Err()
		}
	}))
	done := make(chan *types.JobResult, 1)
	go func() {
		done <- a.AskDirect(types.WithText("start"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	}()
	awaitInteraction(t, events, "json_message")
	a.Pause()
	result := awaitInteractiveResult(t, done)
	if !errors.Is(result.Error, context.Canceled) {
		t.Fatalf("error=%v", result.Error)
	}
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("background dispatcher leaked after cancel")
	}
}

func TestLiveConversationBackpressureAndFinalAdmission(t *testing.T) {
	job := types.NewJob()
	loop := &conversationLoop{job: job, inbox: make(chan openai.ChatCompletionMessage, 1), done: make(chan struct{}), accepting: true, parked: true}
	a := &Agent{options: &options{}, liveConversations: map[string]*conversationLoop{"id": loop}}
	loop.inbox <- openai.ChatCompletionMessage{Role: "user", Content: "already queued"}
	if ok, err := a.InjectChat("id", "overflow", "message"); !ok || !errors.Is(err, ErrConversationBusy) {
		t.Fatalf("backpressure=%v/%v", ok, err)
	}
	if !loop.pending() {
		t.Fatal("queued input was ignored at final guard")
	}
	<-loop.inbox
	if loop.pending() {
		t.Fatal("empty loop did not close admission")
	}
	returned := make(chan bool, 1)
	go func() { ok, _ := a.InjectChat("id", "next turn", "next"); returned <- ok }()
	select {
	case <-returned:
		t.Fatal("new turn admitted before old transcript saved")
	case <-time.After(10 * time.Millisecond):
	}
	close(loop.done)
	select {
	case ok := <-returned:
		if ok {
			t.Fatal("completed loop accepted next turn")
		}
	case <-time.After(time.Second):
		t.Fatal("final admission did not release")
	}
}

func TestLiveConversationTerminalPreservesAcceptedInput(t *testing.T) {
	job := types.NewJob()
	loop := &conversationLoop{job: job, inbox: make(chan openai.ChatCompletionMessage, 2), done: make(chan struct{}), accepting: true, parked: true}
	loop.settled = sync.NewCond(&loop.mu)
	entered, release := make(chan struct{}), make(chan struct{})
	a := &Agent{options: &options{interactionCallback: func(string, any) { close(entered); <-release }}, liveConversations: map[string]*conversationLoop{"id": loop}}
	injected := make(chan error, 1)
	go func() { _, err := a.InjectChat("id", "accepted before failure", "next"); injected <- err }()
	<-entered
	finished := make(chan cogito.Fragment, 1)
	go func() { finished <- loop.finish(cogito.NewEmptyFragment()) }()
	select {
	case <-finished:
		t.Fatal("terminal did not wait for reserved message")
	case <-time.After(10 * time.Millisecond):
	}
	close(release)
	if err := <-injected; err != nil {
		t.Fatal(err)
	}
	select {
	case fragment := <-finished:
		if len(fragment.Messages) != 1 || fragment.Messages[0].Content != "accepted before failure" {
			t.Fatalf("lost accepted input: %+v", fragment.Messages)
		}
	case <-time.After(time.Second):
		t.Fatal("terminal drain blocked")
	}
}

type failingResumedLLM struct {
	*interactiveLLM
	cancel context.CancelFunc
}

func (m failingResumedLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (cogito.LLMReply, cogito.LLMUsage, error) {
	calls, _ := m.counts()
	if calls >= 2 {
		m.cancel()
		return cogito.LLMReply{}, cogito.LLMUsage{}, errors.New("resumed model failed")
	}
	return m.interactiveLLM.CreateChatCompletion(ctx, req)
}

func TestBackgroundDelegationFailurePreservesFollowup(t *testing.T) {
	llm := &interactiveLLM{turns: []interactiveTurn{{"spawn_agent", `{"agent_type":"coder","task":"implement","background":true}`}}}
	a, events := interactiveAgent(t, llm, WithSubAgents(true), WithMaxEvaluationLoops(10), WithSubAgentProvider(func(*types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher) {
		return []cogito.AgentDefinition{{Name: "coder"}}, func(ctx context.Context, _ cogito.AgentRunSpec) (cogito.Fragment, error) {
			<-ctx.Done()
			return cogito.NewEmptyFragment(), ctx.Err()
		}
	}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.llm = failingResumedLLM{llm, cancel}
	done := make(chan *types.JobResult, 1)
	go func() {
		done <- a.AskDirect(types.WithContext(ctx), types.WithText("start"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	}()
	awaitInteraction(t, events, "json_message")
	if handled, err := a.InjectChat("conversation", "retain this followup", "next"); !handled || err != nil {
		t.Fatalf("injection=%v/%v", handled, err)
	}
	result := awaitInteractiveResult(t, done)
	if result.Error == nil {
		t.Fatal("expected resumed model failure")
	}
	history := a.SharedState().ConversationTracker.GetConversation("conversation")
	for _, message := range history {
		if message.Content == "retain this followup" {
			return
		}
	}
	t.Fatalf("failed turn lost accepted followup: %+v", history)
}

func TestDelegatedJobsDoNotCancelPeerConversation(t *testing.T) {
	a, _ := interactiveAgent(t, &interactiveLLM{})
	original := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "shared"}))
	a.currentJobByConversation["shared"] = original
	delegated := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "shared", types.MetadataKeyDelegationID: "child-two"}))
	enqueued := make(chan struct{})
	go func() { a.Enqueue(delegated); close(enqueued) }()
	select {
	case got := <-a.jobQueue:
		if got != delegated {
			t.Fatal("wrong delegated job")
		}
	case <-time.After(time.Second):
		t.Fatal("delegation enqueue blocked")
	}
	<-enqueued
	if original.GetContext().Err() != nil {
		t.Fatal("delegated sibling cancelled peer conversation")
	}
}
