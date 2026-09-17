package webui

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/mudler/LocalAGI/core/agent"
	"github.com/mudler/LocalAGI/core/sse"
	"github.com/mudler/LocalAGI/core/state"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/sashabaranov/go-openai"
)

// A terminating filter exercises the real queue and result lifecycle without an
// external model. Its reply echoes the history delivered to the job.
type chatHistoryFilter struct{}

func (chatHistoryFilter) Name() string    { return "chat-history" }
func (chatHistoryFilter) IsTrigger() bool { return false }
func (chatHistoryFilter) Apply(job *types.Job) (bool, error) {
	history := job.ConversationHistory
	var contents []string
	for _, m := range history {
		contents = append(contents, m.Content)
	}
	reply := strings.Join(contents, "|")
	job.Result.Conversation = append(history, openai.ChatCompletionMessage{Role: "assistant", Content: reply})
	job.Result.SetResponse(reply)
	return false, nil
}

func TestChatHTTPConversationRoundTrip(t *testing.T) {
	pool, err := state.NewAgentPool("test", "", "", "", "", "http://127.0.0.1:1/v1", "", t.TempDir(),
		func(*state.AgentConfig) func(context.Context, *state.AgentPool) []types.Action {
			return func(context.Context, *state.AgentPool) []types.Action { return nil }
		},
		func(*state.AgentConfig) []state.Connector { return nil },
		func(*state.AgentConfig) func(context.Context, *state.AgentPool) []agent.DynamicPrompt {
			return func(context.Context, *state.AgentPool) []agent.DynamicPrompt { return nil }
		},
		func(*state.AgentConfig) types.JobFilters { return types.JobFilters{chatHistoryFilter{}} },
		"5s", false, nil, state.PoolLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.CreateAgent("chat", &state.AgentConfig{Name: "chat"}); err != nil {
		t.Fatal(err)
	}
	defer pool.StopAll()
	client := sse.NewClient("test")
	pool.GetManager("chat").Register(client)
	defer pool.GetManager("chat").Unregister(client.ID())
	app := fiber.New()
	app.Post("/api/chat/:name", (&App{}).Chat(pool))

	post := func(body string, status int) map[string]any {
		t.Helper()
		req := httptest.NewRequest("POST", "/api/chat/chat", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		response, err := app.Test(req)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		if response.StatusCode != status {
			t.Fatalf("status = %d, want %d", response.StatusCode, status)
		}
		var data map[string]any
		if err := json.NewDecoder(response.Body).Decode(&data); err != nil {
			t.Fatal(err)
		}
		return data
	}
	for _, tc := range []struct{ body, conversation, reply string }{
		{`{"message":" first ","conversation_id":"opaque / id"}`, "opaque / id", "first"},
		{`{"message":"second","conversation_id":"opaque / id"}`, "opaque / id", "first|first|second"},
		{`{"message":"other","conversation_id":"other"}`, "other", "other"},
		{`{"message":"anonymous"}`, "", "anonymous"},
		{`{"message":"again"}`, "", "again"},
	} {
		accepted := post(tc.body, fiber.StatusAccepted)
		id := accepted["message_id"]
		var reply string
		completed := false
		// Four events per turn; workers can deliver them in a different order.
		for received := 0; received < 4; {
			select {
			case envelope := <-client.Chan():
				msg := envelope.(*sse.Message)
				if msg.Event != "json_message" && msg.Event != "json_message_status" && msg.Event != "json_error" {
					continue
				}
				received++
				var data map[string]any
				if err := json.Unmarshal([]byte(msg.Data), &data); err != nil {
					t.Fatal(err)
				}
				if data["message_id"] != id {
					t.Fatalf("event message id = %v, want %v", data["message_id"], id)
				}
				if tc.conversation != "" && data["conversation_id"] != tc.conversation {
					t.Fatalf("unscoped event: %s", msg.Data)
				}
				if tc.conversation == "" {
					if _, exists := data["conversation_id"]; exists {
						t.Fatal("anonymous event scoped")
					}
				}
				if data["sender"] == "agent" {
					reply, _ = data["content"].(string)
				}
				if data["status"] == "completed" {
					completed = true
				}
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for chat event")
			}
		}
		if !completed || reply != tc.reply {
			t.Fatalf("reply=%q want=%q completed=%v", reply, tc.reply, completed)
		}
	}
	post(`{"message":"   "}`, fiber.StatusBadRequest)
	post(`{"message":"hello","conversation_id":123}`, fiber.StatusBadRequest)
	post(`{`, fiber.StatusBadRequest)
}
