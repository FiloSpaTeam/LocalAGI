package interactions

import (
	"context"
	"testing"
	"time"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
)

func TestDelegatedInteractionsUseParentSinkAndIdentity(t *testing.T) {
	for _, kind := range []string{"question", "plan"} {
		t.Run(kind, func(t *testing.T) {
			events := make(chan event, 8)
			fallback := make(chan event, 8)
			parent := New(func(name string, payload any) { fallback <- event{name, payload} })
			j := types.NewJob(types.WithUUID("child-message"), types.WithMetadata(map[string]any{
				types.MetadataKeyConversationID: "conversation", types.MetadataKeyDelegationID: "child-id", types.MetadataKeyParentMessageID: "root-message",
			}), types.WithEventCallback(func(name string, payload any) { events <- event{name, payload} }))
			done := make(chan struct{})
			if kind == "question" {
				go func() {
					defer close(done)
					answer, err := parent.HandleQuestion(j, j.GetContext(), question("child-q", "Pick", []string{"yes"}, true, time.Now()))
					if err != nil || answer.Text != "answer" {
						t.Errorf("answer=%+v/%v", answer, err)
					}
				}()
			} else {
				go func() {
					defer close(done)
					decision := parent.ApprovePlan(j, j.GetContext(), &structures.Plan{Description: "child plan", Subtasks: []string{"step"}}, nil)
					if !decision.Approved || decision.Plan == nil || decision.Plan.Subtasks[0] != "edited" {
						t.Errorf("decision=%+v", decision)
					}
				}()
			}
			status := awaitEvent(t, events, "json_message_status")
			assertStatus(t, status.payload, "waiting_user", "conversation", "root-message")
			emitted := awaitEvent(t, events, kind)
			pending := parent.Pending("conversation")
			if len(parent.Pending("other").Questions) != 0 || parent.Pending("other").Plan != nil {
				t.Fatal("interaction leaked across conversations")
			}
			if kind == "question" {
				q := emitted.payload.(Question)
				if q.AgentID != "child-id" || q.MessageID != "root-message" || len(pending.Questions) != 1 {
					t.Fatalf("question=%+v pending=%+v", q, pending)
				}
				if _, handled, err := parent.AnswerText("conversation", "answer"); !handled || err != nil {
					t.Fatalf("answer=%v/%v", handled, err)
				}
			} else {
				p := emitted.payload.(Plan)
				if p.AgentID != "child-id" || p.MessageID != "root-message" || pending.Plan == nil {
					t.Fatalf("plan=%+v pending=%+v", p, pending)
				}
				edited := []string{"edited"}
				if err := parent.Decide(p.ID, true, &edited, ""); err != nil {
					t.Fatal(err)
				}
			}
			assertStatus(t, awaitEvent(t, events, "json_message_status").payload, "processing", "conversation", "root-message")
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("child not resumed")
			}
			if p := parent.Pending("conversation"); len(p.Questions) != 0 || p.Plan != nil {
				t.Fatalf("pending=%+v", p)
			}
			select {
			case e := <-fallback:
				t.Fatalf("event went to fallback: %+v", e)
			default:
			}
		})
	}
}

func TestDelegatedInteractionCancellationClearsParentRegistry(t *testing.T) {
	events := make(chan event, 8)
	parent := New(func(name string, payload any) { events <- event{name, payload} })
	j := job("child-message", "conversation")
	done := make(chan error, 1)
	go func() {
		_, err := parent.HandleQuestion(j, j.GetContext(), question("q", "Pick", []string{"yes"}, false, time.Now()))
		done <- err
	}()
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "question")
	parent.CancelAll()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("missing cancellation")
		}
	case <-time.After(time.Second):
		t.Fatal("cancel blocked")
	}
	if j.GetContext().Err() != context.Canceled {
		t.Fatal("child job not cancelled")
	}
	if len(parent.Pending("conversation").Questions) != 0 {
		t.Fatal("pending after cancel")
	}
	if err := parent.Answer("q", cogito.UserAnswer{Selected: []string{"yes"}}); err != cogito.ErrQuestionNotFound {
		t.Fatalf("late answer=%v", err)
	}
}

func TestAnswerKeepsWaitingStatusForPendingSibling(t *testing.T) {
	events := make(chan event, 12)
	r := New(func(name string, payload any) { events <- event{name, payload} })
	done := make(chan error, 2)
	for _, id := range []string{"one", "two"} {
		j := job(id, "shared")
		go func(id string, j *types.Job) {
			_, err := r.HandleQuestion(j, j.GetContext(), question(id, "Pick", []string{"yes"}, false, time.Now()))
			done <- err
		}(id, j)
		awaitEvent(t, events, "json_message_status")
		awaitEvent(t, events, "question")
	}
	if err := r.Answer("one", cogito.UserAnswer{Selected: []string{"yes"}}); err != nil {
		t.Fatal(err)
	}
	assertStatus(t, awaitEvent(t, events, "json_message_status").payload, "waiting_user", "shared", "one")
	if err := r.Answer("two", cogito.UserAnswer{Selected: []string{"yes"}}); err != nil {
		t.Fatal(err)
	}
	assertStatus(t, awaitEvent(t, events, "json_message_status").payload, "processing", "shared", "two")
	for i := 0; i < 2; i++ {
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			t.Fatal("sibling blocked")
		}
	}
}
