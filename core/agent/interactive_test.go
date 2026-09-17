package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

type interactiveTurn struct{ name, args string }
type interactiveLLM struct {
	mu       sync.Mutex
	turns    []interactiveTurn
	requests []openai.ChatCompletionRequest
	asks     int
}

func (m *interactiveLLM) Ask(ctx context.Context, f cogito.Fragment) (cogito.Fragment, error) {
	if err := ctx.Err(); err != nil {
		return f, err
	}
	m.mu.Lock()
	m.asks++
	m.mu.Unlock()
	return f.AddMessage(cogito.AssistantMessageRole, "finished"), nil
}
func (m *interactiveLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (cogito.LLMReply, cogito.LLMUsage, error) {
	if err := ctx.Err(); err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, req)
	message := openai.ChatCompletionMessage{Role: "assistant", Content: "done"}
	if len(m.turns) > 0 {
		turn := m.turns[0]
		m.turns = m.turns[1:]
		message.Content = ""
		message.ToolCalls = []openai.ToolCall{{ID: fmt.Sprintf("call-%d", len(m.requests)), Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: turn.name, Arguments: turn.args}}}
	}
	return cogito.LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: message}}}}, cogito.LLMUsage{}, nil
}
func (m *interactiveLLM) counts() (int, int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests), m.asks
}

type interactiveObserver struct{}

func (interactiveObserver) NewObservable() *types.Observable { return &types.Observable{} }
func (interactiveObserver) Update(types.Observable)          {}
func (interactiveObserver) History() []types.Observable      { return nil }
func (interactiveObserver) ClearHistory()                    {}

type interactionEvent struct {
	name string
	data map[string]any
}

func interactiveAgent(t *testing.T, llm *interactiveLLM, opts ...Option) (*Agent, <-chan interactionEvent) {
	t.Helper()
	events := make(chan interactionEvent, 30)
	opts = append(opts, WithSchedulerStorePath(filepath.Join(t.TempDir(), "scheduler.json")), WithObserver(interactiveObserver{}), WithInteractionCallback(func(name string, payload any) {
		b, _ := json.Marshal(payload)
		data := map[string]any{}
		_ = json.Unmarshal(b, &data)
		events <- interactionEvent{name, data}
	}))
	a, err := New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	a.llm = llm
	t.Cleanup(a.Stop)
	return a, events
}
func awaitInteraction(t *testing.T, events <-chan interactionEvent, name string) map[string]any {
	t.Helper()
	timer := time.NewTimer(3 * time.Second)
	defer timer.Stop()
	for {
		select {
		case e := <-events:
			if e.name == name {
				return e.data
			}
		case <-timer.C:
			t.Fatalf("no %s event", name)
			return nil
		}
	}
}
func awaitInteractiveResult(t *testing.T, done <-chan *types.JobResult) *types.JobResult {
	t.Helper()
	select {
	case r := <-done:
		if r == nil {
			t.Fatal("nil result")
		}
		return r
	case <-time.After(3 * time.Second):
		t.Fatal("job did not finish")
		return nil
	}
}
func TestInteractiveQuestionToolResumesWithoutExtraCalls(t *testing.T) {
	llm := &interactiveLLM{turns: []interactiveTurn{{"ask_user", `{"question":"Print or validate?","options":["print","validate"]}`}}}
	a, events := interactiveAgent(t, llm, WithUserQuestionsEnabled(true))
	done := make(chan *types.JobResult, 1)
	go func() {
		done <- a.AskDirect(types.WithText("dry run"), types.WithUUID("message"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	}()
	question := awaitInteraction(t, events, "question")
	if question["conversation_id"] != "conversation" || question["message_id"] != "message" {
		t.Fatalf("scope = %v", question)
	}
	calls, asks := llm.counts()
	if calls != 1 || asks != 0 {
		t.Fatalf("waiting model calls = %d/%d", calls, asks)
	}
	if err := a.Interactions().Answer(question["id"].(string), cogito.UserAnswer{Selected: []string{"print"}}); err != nil {
		t.Fatal(err)
	}
	result := awaitInteractiveResult(t, done)
	if result.Error != nil {
		t.Fatal(result.Error)
	}
	found := false
	for _, m := range result.Conversation {
		if m.Role == "tool" && m.Content == "Selected: print" {
			found = true
		}
	}
	if !found {
		t.Fatalf("answer missing from conversation: %+v", result.Conversation)
	}
	calls, asks = llm.counts()
	// LocalAGI keeps the selection loop: the second completion is the reply.
	if calls != 2 || asks != 0 {
		t.Fatalf("model calls = %d/%d", calls, asks)
	}
	if len(a.Interactions().Pending("conversation").Questions) != 0 {
		t.Fatal("answered question remains pending")
	}
}
func TestInteractiveQuestionCancelledOnPauseOrStop(t *testing.T) {
	for _, stop := range []bool{false, true} {
		t.Run(fmt.Sprint(stop), func(t *testing.T) {
			llm := &interactiveLLM{turns: []interactiveTurn{{"ask_user", `{"question":"Continue?"}`}}}
			a, events := interactiveAgent(t, llm, WithUserQuestionsEnabled(true))
			done := make(chan *types.JobResult, 1)
			go func() { done <- a.AskDirect(types.WithText("task")) }()
			awaitInteraction(t, events, "question")
			if stop {
				a.Stop()
			} else {
				a.Pause()
			}
			result := awaitInteractiveResult(t, done)
			if !errors.Is(result.Error, context.Canceled) {
				t.Fatalf("error = %v", result.Error)
			}
			if len(a.Interactions().Pending("").Questions) != 0 {
				t.Fatal("cancelled question pending")
			}
		})
	}
}
func TestInteractiveQuestionsOptIn(t *testing.T) {
	llm := &interactiveLLM{}
	a, _ := interactiveAgent(t, llm)
	result := a.AskDirect(types.WithText("hello"))
	if result.Error != nil {
		t.Fatal(result.Error)
	}
	llm.mu.Lock()
	defer llm.mu.Unlock()
	for _, req := range llm.requests {
		for _, tool := range req.Tools {
			if tool.Function != nil && tool.Function.Name == "ask_user" {
				t.Fatal("questions enabled by default")
			}
		}
	}
}
func TestInteractivePlanRejectionIsSavedAsReply(t *testing.T) {
	llm := &interactiveLLM{turns: []interactiveTurn{{"json", `{"extract_boolean":true}`}, {"json", `{"goal":"task"}`}, {"json", `{"description":"proposal","subtasks":["first"]}`}}}
	a, events := interactiveAgent(t, llm, EnablePlanning, WithRequirePlanApproval(true))
	done := make(chan *types.JobResult, 1)
	go func() {
		done <- a.AskDirect(types.WithText("task"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	}()
	plan := awaitInteraction(t, events, "plan")
	beforeCalls, beforeAsks := llm.counts()
	if err := a.Interactions().Decide(plan["id"].(string), false, nil, ""); err != nil {
		t.Fatal(err)
	}
	result := awaitInteractiveResult(t, done)
	if result.Error != nil || !strings.Contains(strings.ToLower(result.Response), "plan") || !strings.Contains(strings.ToLower(result.Response), "reject") {
		t.Fatalf("result = %+v", result)
	}
	if len(result.Conversation) == 0 || result.Conversation[len(result.Conversation)-1].Role != "assistant" {
		t.Fatal("missing rejection in transcript")
	}
	afterCalls, afterAsks := llm.counts()
	if beforeCalls != afterCalls || beforeAsks != afterAsks {
		t.Fatal("rejection triggered extra model call")
	}
}

type approvalAction struct{ calls int }

func (a *approvalAction) Definition() types.ActionDefinition {
	return types.ActionDefinition{Name: "approval_action", Description: "perform the approved task"}
}
func (a *approvalAction) Plannable() bool { return true }
func (a *approvalAction) Run(context.Context, *types.AgentSharedState, types.ActionParams) (types.ActionResult, error) {
	a.calls++
	return types.ActionResult{Result: "executed"}, nil
}

func TestInteractivePlanApprovalAndRevision(t *testing.T) {
	for _, mode := range []string{"original", "edited", "feedback"} {
		t.Run(mode, func(t *testing.T) {
			turns := []interactiveTurn{{"json", `{"extract_boolean":true}`}, {"json", `{"goal":"task"}`}, {"json", `{"subtasks":["original task"]}`}}
			if mode == "feedback" {
				turns = append(turns, interactiveTurn{"json", `{"subtasks":["revised task"]}`})
			}
			turns = append(turns, interactiveTurn{"approval_action", `{}`}, interactiveTurn{"no_tool_to_call", `{"reasoning":"finished"}`}, interactiveTurn{"json", `{"extract_boolean":true}`})
			llm := &interactiveLLM{turns: turns}
			action := &approvalAction{}
			a, events := interactiveAgent(t, llm, EnablePlanning, WithRequirePlanApproval(true), WithActions(action))
			done := make(chan *types.JobResult, 1)
			go func() { done <- a.AskDirect(types.WithText("task")) }()
			plan := awaitInteraction(t, events, "plan")
			if action.calls != 0 {
				t.Fatal("action executed before approval")
			}
			want := "original task"
			if mode == "feedback" {
				if err := a.Interactions().Decide(plan["id"].(string), false, nil, "revise the task"); err != nil {
					t.Fatal(err)
				}
				next := awaitInteraction(t, events, "plan")
				if next["id"] == plan["id"] {
					t.Fatal("revised plan reused old approval id")
				}
				plan = next
				want = "revised task"
			}
			var subtasks *[]string
			if mode == "edited" {
				edited := []string{"edited task"}
				subtasks = &edited
				want = "edited task"
			}
			if err := a.Interactions().Decide(plan["id"].(string), true, subtasks, ""); err != nil {
				t.Fatal(err)
			}
			result := awaitInteractiveResult(t, done)
			if result.Error != nil {
				t.Fatal(result.Error)
			}
			if action.calls != 1 {
				t.Fatalf("executions = %d", action.calls)
			}
			if len(result.Plans) != 1 || len(result.Plans[0].Plan.Subtasks) != 1 || result.Plans[0].Plan.Subtasks[0] != want {
				t.Fatalf("executed plans = %+v", result.Plans)
			}
		})
	}
}

func TestInteractiveNewJobCancelsPreviousQuestion(t *testing.T) {
	llm := &interactiveLLM{turns: []interactiveTurn{{"ask_user", `{"question":"Continue?"}`}}}
	a, events := interactiveAgent(t, llm, WithUserQuestionsEnabled(true))
	go func() { _ = a.Run() }()
	first, second := make(chan *types.JobResult, 1), make(chan *types.JobResult, 1)
	opts := []types.JobOption{types.WithText("task"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "same"})}
	go func() { first <- a.Ask(opts...) }()
	awaitInteraction(t, events, "question")
	go func() {
		second <- a.Ask(types.WithText("replace task"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "same"}))
	}()
	if result := awaitInteractiveResult(t, first); !errors.Is(result.Error, context.Canceled) {
		t.Fatalf("first error = %v", result.Error)
	}
	if result := awaitInteractiveResult(t, second); result.Error != nil {
		t.Fatalf("second error = %v", result.Error)
	}
	if len(a.Interactions().Pending("same").Questions) != 0 {
		t.Fatal("cancelled question retained")
	}
}

func TestInteractivePlanApprovalRequiresPlanning(t *testing.T) {
	llm := &interactiveLLM{}
	a, events := interactiveAgent(t, llm, WithRequirePlanApproval(true))
	if result := a.AskDirect(types.WithText("hello")); result.Error != nil {
		t.Fatal(result.Error)
	}
	select {
	case event := <-events:
		t.Fatalf("unexpected interaction: %s", event.name)
	default:
	}
}
