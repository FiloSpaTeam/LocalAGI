package webui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/mudler/LocalAGI/core/agent"
	coreInteractions "github.com/mudler/LocalAGI/core/interactions"
	"github.com/mudler/LocalAGI/core/state"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
)

func newInteractionTestPool(t *testing.T) *state.AgentPool {
	t.Helper()
	pool, err := state.NewAgentPool("test", "", "", "", "", "http://127.0.0.1:1/v1", "", t.TempDir(),
		func(*state.AgentConfig) func(context.Context, *state.AgentPool) []types.Action {
			return func(context.Context, *state.AgentPool) []types.Action { return nil }
		},
		func(*state.AgentConfig) []state.Connector { return nil },
		func(*state.AgentConfig) func(context.Context, *state.AgentPool) []agent.DynamicPrompt {
			return func(context.Context, *state.AgentPool) []agent.DynamicPrompt { return nil }
		},
		func(*state.AgentConfig) types.JobFilters { return nil },
		"5s", false, nil, state.PoolLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.CreateAgent("chat", &state.AgentConfig{Name: "chat"}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.StopAll)
	return pool
}

func newInteractionTestApp(pool *state.AgentPool) *fiber.App {
	app := fiber.New()
	h := &App{}
	app.Post("/api/chat/:name/answer", h.AnswerInteraction(pool))
	app.Post("/api/chat/:name/plan", h.DecidePlan(pool))
	app.Get("/api/chat/:name/pending", h.PendingInteractions(pool))
	return app
}

func interactionRequest(t *testing.T, app *fiber.App, method, path, body string) (*http.Response, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	response, err := app.Test(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	var data map[string]any
	if err := json.NewDecoder(response.Body).Decode(&data); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return response, data
}

func TestInteractionHandlersRejectUnknownAgentAndInvalidPayloads(t *testing.T) {
	pool := newInteractionTestPool(t)
	app := newInteractionTestApp(pool)

	tests := []struct {
		method string
		path   string
		body   string
		status int
	}{
		{http.MethodPost, "/api/chat/missing/answer", `{"question_id":"q","text":"yes"}`, fiber.StatusNotFound},
		{http.MethodPost, "/api/chat/chat/answer", `{`, fiber.StatusBadRequest},
		{http.MethodPost, "/api/chat/chat/answer", `{"text":"yes"}`, fiber.StatusBadRequest},
		{http.MethodPost, "/api/chat/chat/answer", `{"question_id":"missing","text":"yes"}`, fiber.StatusNotFound},
		{http.MethodPost, "/api/chat/chat/plan", `{`, fiber.StatusBadRequest},
		{http.MethodPost, "/api/chat/chat/plan", `{"approved":true}`, fiber.StatusBadRequest},
		{http.MethodPost, "/api/chat/chat/plan", `{"plan_id":"p"}`, fiber.StatusBadRequest},
		{http.MethodPost, "/api/chat/chat/plan", `{"plan_id":"missing","approved":true}`, fiber.StatusNotFound},
		{http.MethodGet, "/api/chat/missing/pending", "", fiber.StatusNotFound},
	}
	for _, tc := range tests {
		response, _ := interactionRequest(t, app, tc.method, tc.path, tc.body)
		if response.StatusCode != tc.status {
			t.Errorf("%s %s status = %d, want %d", tc.method, tc.path, response.StatusCode, tc.status)
		}
	}
}

func TestAnswerInteractionResumesQuestionAndClearsPending(t *testing.T) {
	pool := newInteractionTestPool(t)
	app := newInteractionTestApp(pool)
	registry := pool.GetAgent("chat").Interactions()
	job := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation-a"}))
	answered := make(chan cogito.UserAnswer, 1)
	go func() {
		answer, err := registry.HandleQuestion(job, context.Background(), cogito.UserQuestion{
			ID: "question-1", Question: "Pick one", Options: []string{"alpha", "beta"}, AskedAt: time.Now(),
		})
		if err == nil {
			answered <- answer
		}
	}()
	waitForPendingQuestion(t, registry, "conversation-a", "question-1")

	response, data := interactionRequest(t, app, http.MethodPost, "/api/chat/chat/answer", `{"question_id":"question-1","selected":["beta"]}`)
	if response.StatusCode != fiber.StatusOK || data["status"] != "answer_received" {
		t.Fatalf("response status=%d body=%v", response.StatusCode, data)
	}
	select {
	case answer := <-answered:
		if len(answer.Selected) != 1 || answer.Selected[0] != "beta" {
			t.Fatalf("answer = %#v", answer)
		}
	case <-time.After(time.Second):
		t.Fatal("question was not resumed")
	}
	response, data = interactionRequest(t, app, http.MethodGet, "/api/chat/chat/pending?conversation_id=conversation-a", "")
	if response.StatusCode != fiber.StatusOK {
		t.Fatalf("pending status = %d", response.StatusCode)
	}
	questions, ok := data["questions"].([]any)
	if !ok || len(questions) != 0 {
		t.Fatalf("pending questions = %#v", data["questions"])
	}
}

func TestInteractionHandlersReturnBadRequestWithoutClearingInvalidInput(t *testing.T) {
	pool := newInteractionTestPool(t)
	app := newInteractionTestApp(pool)
	registry := pool.GetAgent("chat").Interactions()

	questionJob := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	go registry.HandleQuestion(questionJob, context.Background(), cogito.UserQuestion{
		ID: "restricted", Question: "Pick", Options: []string{"yes"}, AskedAt: time.Now(),
	})
	waitForPendingQuestion(t, registry, "conversation", "restricted")
	response, _ := interactionRequest(t, app, http.MethodPost, "/api/chat/chat/answer", `{"question_id":"restricted","text":"yes"}`)
	if response.StatusCode != fiber.StatusBadRequest || len(registry.Pending("conversation").Questions) != 1 {
		t.Fatalf("invalid answer status=%d pending=%#v", response.StatusCode, registry.Pending("conversation"))
	}

	planJob := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	go registry.ApprovePlan(planJob, context.Background(), &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
	deadline := time.Now().Add(time.Second)
	var planID string
	for time.Now().Before(deadline) {
		if plan := registry.Pending("conversation").Plan; plan != nil {
			planID = plan.ID
			break
		}
		time.Sleep(time.Millisecond)
	}
	if planID == "" {
		t.Fatal("plan did not become pending")
	}
	response, _ = interactionRequest(t, app, http.MethodPost, "/api/chat/chat/plan", `{"plan_id":"`+planID+`","approved":true,"subtasks":[]}`)
	if response.StatusCode != fiber.StatusBadRequest || registry.Pending("conversation").Plan == nil {
		t.Fatalf("invalid plan status=%d pending=%#v", response.StatusCode, registry.Pending("conversation"))
	}
	registry.CancelAll()
}

func waitForPendingQuestion(t *testing.T, registry interface {
	Pending(string) coreInteractions.Snapshot
}, conversationID, questionID string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		questions := registry.Pending(conversationID).Questions
		if len(questions) == 1 && questions[0].ID == questionID {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("question did not become pending")
}

func TestPendingInteractionsFiltersExactConversationIncludingEmpty(t *testing.T) {
	pool := newInteractionTestPool(t)
	app := newInteractionTestApp(pool)
	registry := pool.GetAgent("chat").Interactions()
	for _, item := range []struct{ id, conversation string }{{"empty", ""}, {"one", "conversation-1"}, {"two", "conversation-2"}} {
		item := item
		job := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: item.conversation}))
		go registry.HandleQuestion(job, context.Background(), cogito.UserQuestion{ID: item.id, Question: item.id, AllowFreeText: true, AskedAt: time.Now()})
		waitForPendingQuestion(t, registry, item.conversation, item.id)
	}

	for _, tc := range []struct{ path, want string }{
		{"/api/chat/chat/pending", "empty"},
		{"/api/chat/chat/pending?conversation_id=conversation-1", "one"},
		{"/api/chat/chat/pending?conversation_id=conversation-2", "two"},
	} {
		response, data := interactionRequest(t, app, http.MethodGet, tc.path, "")
		questions, ok := data["questions"].([]any)
		if response.StatusCode != fiber.StatusOK || !ok || len(questions) != 1 {
			t.Fatalf("GET %s status=%d questions=%#v", tc.path, response.StatusCode, data["questions"])
		}
		question := questions[0].(map[string]any)
		if question["id"] != tc.want {
			t.Fatalf("GET %s question id=%v, want %s", tc.path, question["id"], tc.want)
		}
	}
	registry.CancelAll()
}

func TestDecidePlanResumesEditedAndRejectedPlans(t *testing.T) {
	tests := []struct {
		name string
		body func(string) string
		want cogito.PlanDecision
	}{
		{
			name: "edited",
			body: func(id string) string {
				return `{"plan_id":"` + id + `","approved":true,"subtasks":["edited one","edited two"]}`
			},
			want: cogito.PlanDecision{Approved: true, Plan: &structures.Plan{Description: "proposal", Subtasks: []string{"edited one", "edited two"}}},
		},
		{
			name: "rejected",
			body: func(id string) string { return `{"plan_id":"` + id + `","approved":false}` },
			want: cogito.PlanDecision{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pool := newInteractionTestPool(t)
			app := newInteractionTestApp(pool)
			registry := pool.GetAgent("chat").Interactions()
			job := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
			decided := make(chan cogito.PlanDecision, 1)
			go func() {
				decided <- registry.ApprovePlan(job, context.Background(), &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
			}()
			deadline := time.Now().Add(time.Second)
			var planID string
			for time.Now().Before(deadline) {
				if plan := registry.Pending("conversation").Plan; plan != nil {
					planID = plan.ID
					break
				}
				time.Sleep(time.Millisecond)
			}
			if planID == "" {
				t.Fatal("plan did not become pending")
			}

			response, data := interactionRequest(t, app, http.MethodPost, "/api/chat/chat/plan", tc.body(planID))
			if response.StatusCode != fiber.StatusOK || data["status"] != "plan_decision_received" {
				t.Fatalf("response status=%d body=%v", response.StatusCode, data)
			}
			select {
			case got := <-decided:
				gotJSON, _ := json.Marshal(got)
				wantJSON, _ := json.Marshal(tc.want)
				if string(gotJSON) != string(wantJSON) {
					t.Fatalf("decision=%s want=%s", gotJSON, wantJSON)
				}
			case <-time.After(time.Second):
				t.Fatal("plan was not resumed")
			}
		})
	}
}

func TestChatRoutesFreeTextToPendingQuestion(t *testing.T) {
	pool := newInteractionTestPool(t)
	app := fiber.New()
	app.Post("/api/chat/:name", (&App{}).Chat(pool))
	registry := pool.GetAgent("chat").Interactions()
	job := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	answered := make(chan cogito.UserAnswer, 1)
	go func() {
		answer, _ := registry.HandleQuestion(job, context.Background(), cogito.UserQuestion{
			ID: "free-text", Question: "Tell me", AllowFreeText: true, AskedAt: time.Now(),
		})
		answered <- answer
	}()
	waitForPendingQuestion(t, registry, "conversation", "free-text")

	response, data := interactionRequest(t, app, http.MethodPost, "/api/chat/chat", `{"message":"the answer","conversation_id":"conversation"}`)
	if response.StatusCode != fiber.StatusAccepted || data["status"] != "answer_received" || data["question_id"] != "free-text" {
		t.Fatalf("response status=%d body=%v", response.StatusCode, data)
	}
	select {
	case answer := <-answered:
		if answer.Text != "the answer" {
			t.Fatalf("answer=%#v", answer)
		}
	case <-time.After(time.Second):
		t.Fatal("chat answer did not resume question")
	}
}

func TestChatConflictsWhenPendingQuestionForbidsFreeText(t *testing.T) {
	pool := newInteractionTestPool(t)
	app := fiber.New()
	app.Post("/api/chat/:name", (&App{}).Chat(pool))
	registry := pool.GetAgent("chat").Interactions()
	job := types.NewJob(types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
	go registry.HandleQuestion(job, context.Background(), cogito.UserQuestion{
		ID: "choice", Question: "Pick", Options: []string{"yes"}, AskedAt: time.Now(),
	})
	waitForPendingQuestion(t, registry, "conversation", "choice")

	response, data := interactionRequest(t, app, http.MethodPost, "/api/chat/chat", `{"message":"yes","conversation_id":"conversation"}`)
	if response.StatusCode != fiber.StatusConflict || data["pending_question_id"] != "choice" {
		t.Fatalf("response status=%d body=%v", response.StatusCode, data)
	}
	if pending := registry.Pending("conversation"); len(pending.Questions) != 1 || pending.Questions[0].ID != "choice" {
		t.Fatalf("pending=%#v", pending)
	}
	registry.CancelAll()
}
