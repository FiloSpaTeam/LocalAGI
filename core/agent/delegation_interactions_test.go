package agent

import (
	"context"
	"testing"
	"time"

	"github.com/mudler/LocalAGI/core/interactions"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
)

func TestPoolChildInteractionsBelongToParent(t *testing.T) {
	for _, kind := range []string{"question", "plan", "cancel-question", "silent-root"} {
		t.Run(kind, func(t *testing.T) {
			turns := []interactiveTurn{{"ask_user", `{"question":"Choose","options":["yes"],"allow_free_text":true}`}}
			options := []Option{WithUserQuestionsEnabled(true)}
			if kind == "plan" {
				turns = []interactiveTurn{{"json", `{"extract_boolean":true}`}, {"json", `{"goal":"task"}`}, {"json", `{"subtasks":["child step"]}`}}
				options = []Option{EnablePlanning, WithRequirePlanApproval(true)}
			}
			child, childEvents := interactiveAgent(t, &interactiveLLM{turns: turns}, options...)
			parentLLM := &interactiveLLM{turns: []interactiveTurn{{"spawn_agent", `{"agent_type":"child","task":"task","background":true}`}}}
			parent, events := interactiveAgent(t, parentLLM, WithSubAgents(true), WithMaxEvaluationLoops(10), WithSubAgentProvider(func(job *types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher) {
				return []cogito.AgentDefinition{{Name: "child"}}, func(ctx context.Context, spec cogito.AgentRunSpec) (cogito.Fragment, error) {
					result := child.AskDirect(types.WithText(spec.Task), types.WithContext(ctx), types.WithInteractionHandler(job.InteractionHandler), types.WithEventCallback(job.EventCallback), types.WithMetadata(map[string]any{
						types.MetadataKeyConversationID: job.Metadata[types.MetadataKeyConversationID], types.MetadataKeyParentMessageID: job.UUID, types.MetadataKeyDelegationID: spec.ID,
					}))
					return cogito.NewEmptyFragment().AddMessage("assistant", result.Response), result.Error
				}
			}))
			if kind == "silent-root" {
				parent.options.interactionCallback = nil
				parent.interactions = interactions.New(nil)
			}
			done := make(chan *types.JobResult, 1)
			go func() {
				done <- parent.AskDirect(types.WithUUID("root"), types.WithText("start"), types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation"}))
			}()
			eventName := "question"
			if kind == "plan" {
				eventName = "plan"
			}
			var shown map[string]any
			if kind == "silent-root" {
				deadline := time.Now().Add(time.Second)
				for {
					pending := parent.Interactions().Pending("conversation")
					if len(pending.Questions) > 0 {
						q := pending.Questions[0]
						shown = map[string]any{"message_id": q.MessageID, "conversation_id": q.ConversationID, "agent_id": q.AgentID}
						break
					}
					if time.Now().After(deadline) {
						t.Fatal("silent root did not retain question")
					}
					time.Sleep(time.Millisecond)
				}
			} else {
				shown = awaitInteraction(t, events, eventName)
			}
			if shown["message_id"] != "root" || shown["conversation_id"] != "conversation" || shown["agent_id"] == "" {
				t.Fatalf("unrouted child interaction: %v", shown)
			}
			if childPending := child.Interactions().Pending("conversation"); len(childPending.Questions) > 0 || childPending.Plan != nil {
				t.Fatal("interaction belongs to child registry")
			}
			pending := parent.Interactions().Pending("conversation")
			if kind == "plan" {
				if pending.Plan == nil {
					t.Fatal("parent missing plan")
				}
				if err := parent.Interactions().Decide(pending.Plan.ID, false, nil, ""); err != nil {
					t.Fatal(err)
				}
			} else {
				if len(pending.Questions) != 1 {
					t.Fatal("parent missing question")
				}
				if kind == "cancel-question" {
					parent.Pause()
				} else if _, handled, err := parent.Interactions().AnswerText("conversation", "yes"); !handled || err != nil {
					t.Fatalf("answer=%v/%v", handled, err)
				}
			}
			result := awaitInteractiveResult(t, done)
			if kind != "cancel-question" && result.Error != nil {
				t.Fatal(result.Error)
			}
			deadline := time.Now().Add(time.Second)
			for {
				p := parent.Interactions().Pending("conversation")
				if len(p.Questions) == 0 && p.Plan == nil {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("parent retains child interaction after completion")
				}
				time.Sleep(time.Millisecond)
			}
			select {
			case leaked := <-childEvents:
				t.Fatalf("child event leaked to peer sink: %+v", leaked)
			default:
			}
		})
	}
}
