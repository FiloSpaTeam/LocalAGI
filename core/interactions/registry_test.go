package interactions

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
)

type event struct {
	name    string
	payload any
}

func question(id, text string, options []string, freeText bool, askedAt time.Time) cogito.UserQuestion {
	return cogito.UserQuestion{ID: id, Question: text, Options: options, AllowFreeText: freeText, AskedAt: askedAt}
}

func job(messageID, conversationID string) *types.Job {
	return types.NewJob(
		types.WithUUID(messageID),
		types.WithMetadata(map[string]any{types.MetadataKeyConversationID: conversationID}),
	)
}

func awaitEvent(t *testing.T, ch <-chan event, name string) event {
	t.Helper()
	select {
	case got := <-ch:
		if got.name != name {
			t.Fatalf("event = %q, want %q", got.name, name)
		}
		return got
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for %q", name)
		return event{}
	}
}

func assertStatus(t *testing.T, payload any, status, conversationID, messageID string) {
	t.Helper()
	got := payload.(map[string]any)
	if got["status"] != status || got["conversation_id"] != conversationID || got["message_id"] != messageID {
		t.Fatalf("status payload = %#v", got)
	}
	timestamp, ok := got["timestamp"].(string)
	if !ok {
		t.Fatalf("status timestamp = %#v", got["timestamp"])
	}
	if _, err := time.Parse(time.RFC3339, timestamp); err != nil {
		t.Fatalf("status timestamp %q: %v", timestamp, err)
	}
}

func TestHandleQuestionEmitsScopedSnapshotAndAnswers(t *testing.T) {
	events := make(chan event, 4)
	r := New(func(name string, payload any) { events <- event{name, payload} })
	askedAt := time.Date(2026, 9, 17, 10, 0, 0, 0, time.UTC)
	q := question("question-1", "Pick one", []string{"A", "B"}, false, askedAt)
	q.Options = append(q.Options, "C")
	j := job("message-1", "conversation-1")
	result := make(chan cogito.UserAnswer, 1)
	errResult := make(chan error, 1)
	go func() {
		answer, err := r.HandleQuestion(j, j.GetContext(), q)
		result <- answer
		errResult <- err
	}()

	status := awaitEvent(t, events, "json_message_status")
	assertStatus(t, status.payload, "waiting_user", "conversation-1", "message-1")
	emitted := awaitEvent(t, events, "question").payload.(Question)
	if emitted.ID != "question-1" || emitted.ConversationID != "conversation-1" || emitted.MessageID != "message-1" || !emitted.Timestamp.Equal(askedAt) {
		t.Fatalf("question event = %#v", emitted)
	}

	snapshot := r.Pending("conversation-1")
	if len(snapshot.Questions) != 1 || snapshot.Plan != nil || !reflect.DeepEqual(snapshot.Questions[0].Options, []string{"A", "B", "C"}) {
		t.Fatalf("pending = %#v", snapshot)
	}
	snapshot.Questions[0].Options[0] = "mutated"
	if got := r.Pending("conversation-1").Questions[0].Options[0]; got != "A" {
		t.Fatalf("pending snapshot aliases state: %q", got)
	}

	if err := r.Answer("question-1", cogito.UserAnswer{Selected: []string{"B"}}); err != nil {
		t.Fatal(err)
	}
	processing := awaitEvent(t, events, "json_message_status")
	assertStatus(t, processing.payload, "processing", "conversation-1", "message-1")
	if got := <-result; !reflect.DeepEqual(got, cogito.UserAnswer{Selected: []string{"B"}}) {
		t.Fatalf("answer = %#v", got)
	}
	if err := <-errResult; err != nil {
		t.Fatal(err)
	}
	if got := r.Pending("conversation-1"); len(got.Questions) != 0 || got.Plan != nil {
		t.Fatalf("answered item remains pending: %#v", got)
	}
	if err := r.Answer("question-1", cogito.UserAnswer{Selected: []string{"B"}}); !errors.Is(err, cogito.ErrQuestionNotFound) {
		t.Fatalf("repeat answer error = %v", err)
	}
}

func TestQuestionValidationDoesNotReleaseHandler(t *testing.T) {
	events := make(chan event, 8)
	r := New(func(name string, payload any) { events <- event{name, payload} })
	j := job("message", "conversation")
	result := make(chan error, 1)
	go func() {
		_, err := r.HandleQuestion(j, j.GetContext(), question("q", "Pick", []string{"yes"}, false, time.Now()))
		result <- err
	}()
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "question")

	if err := r.Answer("q", cogito.UserAnswer{Text: "anything"}); !errors.Is(err, cogito.ErrInvalidAnswer) {
		t.Fatalf("invalid answer error = %v", err)
	}
	select {
	case got := <-events:
		t.Fatalf("invalid answer emitted event %#v", got)
	default:
	}
	if len(r.Pending("conversation").Questions) != 1 {
		t.Fatal("invalid answer removed pending question")
	}

	j.Cancel()
	if err := <-result; !errors.Is(err, cogito.ErrQuestionCancelled) {
		t.Fatalf("cancellation error = %v", err)
	}
}

func TestQuestionEventMayAnswerReentrantlyWithoutStaleStatus(t *testing.T) {
	var r *Registry
	var mu sync.Mutex
	var names []string
	var answerErr error
	r = New(func(name string, payload any) {
		mu.Lock()
		names = append(names, name)
		mu.Unlock()
		if name == "question" {
			answerErr = r.Answer(payload.(Question).ID, cogito.UserAnswer{Text: "now"})
		}
	})
	j := job("message", "conversation")
	answer, err := r.HandleQuestion(j, j.GetContext(), question("q", "Tell me", nil, true, time.Now()))
	if err != nil || answer.Text != "now" || answerErr != nil {
		t.Fatalf("answer = %#v, handle err = %v, answer err = %v", answer, err, answerErr)
	}
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(names, []string{"json_message_status", "question", "json_message_status"}) {
		t.Fatalf("events = %v", names)
	}
}

func TestQuestionIsNotActionableUntilWaitingStatusIsDelivered(t *testing.T) {
	statusStarted := make(chan struct{})
	releaseStatus := make(chan struct{})
	events := make(chan event, 4)
	r := New(func(name string, payload any) {
		if name == "json_message_status" && payload.(map[string]any)["status"] == "waiting_user" {
			close(statusStarted)
			<-releaseStatus
		}
		events <- event{name, payload}
	})
	j := job("message", "conversation")
	done := make(chan error, 1)
	go func() {
		_, err := r.HandleQuestion(j, j.GetContext(), question("q", "Tell me", nil, true, time.Now()))
		done <- err
	}()
	<-statusStarted
	pending := r.Pending("conversation")
	prematureErr := r.Answer("q", cogito.UserAnswer{Text: "too soon"})
	close(releaseStatus)

	if len(pending.Questions) != 0 {
		t.Fatalf("question published before waiting status returned: %#v", pending)
	}
	if !errors.Is(prematureErr, cogito.ErrQuestionNotFound) {
		t.Fatalf("premature answer error = %v", prematureErr)
	}
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "question")
	r.CancelAll()
	if err := <-done; !errors.Is(err, cogito.ErrQuestionCancelled) {
		t.Fatalf("handle error = %v", err)
	}
}

func TestQuestionAnswerWaitsUntilQuestionEventReturns(t *testing.T) {
	cardStarted := make(chan struct{})
	releaseCard := make(chan struct{})
	events := make(chan event, 8)
	r := New(func(name string, payload any) {
		if name == "question" {
			close(cardStarted)
			<-releaseCard
		}
		events <- event{name, payload}
	})
	j := job("message", "conversation")
	done := make(chan cogito.UserAnswer, 1)
	go func() {
		answer, _ := r.HandleQuestion(j, j.GetContext(), question("q", "Tell me", nil, true, time.Now()))
		done <- answer
	}()
	awaitEvent(t, events, "json_message_status")
	<-cardStarted
	if len(r.Pending("conversation").Questions) != 1 {
		t.Fatal("question is not reservable while its event is being delivered")
	}
	if err := r.Answer("q", cogito.UserAnswer{Text: "answer"}); err != nil {
		t.Fatal(err)
	}
	select {
	case got := <-events:
		t.Fatalf("event arrived before question callback returned: %#v", got)
	default:
	}
	select {
	case got := <-done:
		t.Fatalf("handler resumed before question callback returned: %#v", got)
	default:
	}
	close(releaseCard)
	awaitEvent(t, events, "question")
	awaitEvent(t, events, "json_message_status")
	if got := <-done; got.Text != "answer" {
		t.Fatalf("answer = %#v", got)
	}
}

func TestAnswerTextRoutesOldestQuestionOnlyWithinConversation(t *testing.T) {
	events := make(chan event, 20)
	r := New(func(name string, payload any) { events <- event{name, payload} })
	base := time.Now()
	start := func(j *types.Job, q cogito.UserQuestion) chan cogito.UserAnswer {
		out := make(chan cogito.UserAnswer, 1)
		go func() { answer, _ := r.HandleQuestion(j, j.GetContext(), q); out <- answer }()
		awaitEvent(t, events, "json_message_status")
		awaitEvent(t, events, "question")
		return out
	}
	one := start(job("m1", "same"), question("newer", "Newer", nil, true, base.Add(time.Second)))
	two := start(job("m2", "other"), question("other", "Other", nil, true, base.Add(-time.Second)))
	three := start(job("m3", "same"), question("oldest", "Oldest", nil, true, base))

	id, handled, err := r.AnswerText("", "answer")
	if id != "" || handled || err != nil {
		t.Fatalf("empty conversation result = %q, %v, %v", id, handled, err)
	}
	id, handled, err = r.AnswerText("same", "answer")
	if id != "oldest" || !handled || err != nil {
		t.Fatalf("routed result = %q, %v, %v", id, handled, err)
	}
	awaitEvent(t, events, "json_message_status")
	if got := <-three; got.Text != "answer" {
		t.Fatalf("answer = %#v", got)
	}
	select {
	case <-one:
		t.Fatal("answered newer question")
	case <-two:
		t.Fatal("answered other conversation")
	default:
	}

	restricted := start(job("m4", "restricted"), question("restricted", "Pick", []string{"yes"}, false, base))
	id, handled, err = r.AnswerText("restricted", "yes")
	if id != "restricted" || !handled || !errors.Is(err, ErrFreeTextNotAllowed) {
		t.Fatalf("restricted result = %q, %v, %v", id, handled, err)
	}
	select {
	case <-restricted:
		t.Fatal("free text released restricted question")
	default:
	}

	if id, handled, err = r.AnswerText("missing", "answer"); id != "" || handled || err != nil {
		t.Fatalf("missing result = %q, %v, %v", id, handled, err)
	}
	r.CancelAll()
}

func TestCancellationRemovesQuestionsAndRejectsStaleAnswers(t *testing.T) {
	events := make(chan event, 4)
	r := New(func(name string, payload any) { events <- event{name, payload} })
	j := job("message", "conversation")
	done := make(chan error, 1)
	go func() {
		_, err := r.HandleQuestion(j, j.GetContext(), question("q", "Tell", nil, true, time.Now()))
		done <- err
	}()
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "question")
	j.Cancel()
	if err := <-done; !errors.Is(err, cogito.ErrQuestionCancelled) {
		t.Fatalf("handle error = %v", err)
	}
	if len(r.Pending("conversation").Questions) != 0 {
		t.Fatal("cancelled question remains pending")
	}
	if err := r.Answer("q", cogito.UserAnswer{Text: "late"}); !errors.Is(err, cogito.ErrQuestionNotFound) {
		t.Fatalf("stale answer error = %v", err)
	}
}

func TestApprovePlanDecisionsAndCopies(t *testing.T) {
	tests := []struct {
		name     string
		approved bool
		edits    *[]string
		feedback string
		want     cogito.PlanDecision
	}{
		{name: "approve extracted", approved: true, want: cogito.PlanDecision{Approved: true}},
		{name: "approve edits", approved: true, edits: &[]string{"edited one", "edited two"}, want: cogito.PlanDecision{Approved: true, Plan: &structures.Plan{Description: "proposal", Subtasks: []string{"edited one", "edited two"}}}},
		{name: "reject", want: cogito.PlanDecision{}},
		{name: "request revision", feedback: "split the second step", want: cogito.PlanDecision{Feedback: "split the second step"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := make(chan event, 4)
			r := New(func(name string, payload any) { events <- event{name, payload} })
			j := job("message", "conversation")
			original := &structures.Plan{Description: "proposal", Subtasks: []string{"one", "two"}}
			result := make(chan cogito.PlanDecision, 1)
			go func() { result <- r.ApprovePlan(j, j.GetContext(), original, &structures.Goal{Goal: "goal"}) }()
			awaitEvent(t, events, "json_message_status")
			shown := awaitEvent(t, events, "plan").payload.(Plan)
			if shown.ID == "" || shown.ConversationID != "conversation" || shown.MessageID != "message" || shown.Description != "proposal" || !reflect.DeepEqual(shown.Subtasks, []string{"one", "two"}) || shown.Timestamp.IsZero() {
				t.Fatalf("plan event = %#v", shown)
			}
			snapshot := r.Pending("conversation")
			if snapshot.Plan == nil || snapshot.Plan.ID != shown.ID || len(snapshot.Questions) != 0 {
				t.Fatalf("pending = %#v", snapshot)
			}
			snapshot.Plan.Subtasks[0] = "mutated"
			if got := r.Pending("conversation").Plan.Subtasks[0]; got != "one" {
				t.Fatalf("snapshot aliases plan: %q", got)
			}
			if err := r.Decide(shown.ID, tt.approved, tt.edits, tt.feedback); err != nil {
				t.Fatal(err)
			}
			awaitEvent(t, events, "json_message_status")
			if got := <-result; !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("decision = %#v, want %#v", got, tt.want)
			}
			if r.Pending("conversation").Plan != nil {
				t.Fatal("decided plan remains pending")
			}
			if !reflect.DeepEqual(original.Subtasks, []string{"one", "two"}) {
				t.Fatalf("original plan mutated: %#v", original)
			}
			if err := r.Decide(shown.ID, true, nil, ""); !errors.Is(err, ErrPlanNotFound) {
				t.Fatalf("repeat decision error = %v", err)
			}
		})
	}
}

func TestDecideValidatesEditedPlan(t *testing.T) {
	for _, edits := range [][]string{nil, {}, {"ok", "  "}} {
		events := make(chan event, 4)
		r := New(func(name string, payload any) { events <- event{name, payload} })
		j := job("message", "conversation")
		go r.ApprovePlan(j, j.GetContext(), &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
		awaitEvent(t, events, "json_message_status")
		shown := awaitEvent(t, events, "plan").payload.(Plan)
		if err := r.Decide(shown.ID, true, &edits, ""); !errors.Is(err, ErrInvalidPlanDecision) {
			t.Fatalf("edits %#v: error = %v", edits, err)
		}
		if r.Pending("conversation").Plan == nil {
			t.Fatalf("edits %#v removed pending plan", edits)
		}
		r.CancelAll()
	}
}

func TestPlanIsNotActionableUntilWaitingStatusIsDelivered(t *testing.T) {
	statusStarted := make(chan struct{})
	releaseStatus := make(chan struct{})
	events := make(chan event, 4)
	r := New(func(name string, payload any) {
		if name == "json_message_status" && payload.(map[string]any)["status"] == "waiting_user" {
			close(statusStarted)
			<-releaseStatus
		}
		events <- event{name, payload}
	})
	j := job("message", "conversation")
	done := make(chan cogito.PlanDecision, 1)
	go func() {
		done <- r.ApprovePlan(j, j.GetContext(), &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
	}()
	<-statusStarted
	pending := r.Pending("conversation")
	var planID string
	r.mu.Lock()
	for id := range r.plans {
		planID = id
	}
	r.mu.Unlock()
	prematureErr := r.Decide(planID, true, nil, "")
	close(releaseStatus)

	if pending.Plan != nil {
		t.Fatalf("plan published before waiting status returned: %#v", pending)
	}
	if !errors.Is(prematureErr, ErrPlanNotFound) {
		t.Fatalf("premature decision error = %v", prematureErr)
	}
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "plan")
	r.CancelAll()
	if got := <-done; got.Approved || got.Plan != nil || got.Feedback != "" {
		t.Fatalf("cancelled plan decision = %#v", got)
	}
}

func TestPlanDecisionWaitsUntilPlanEventReturns(t *testing.T) {
	cardStarted := make(chan struct{})
	releaseCard := make(chan struct{})
	events := make(chan event, 8)
	r := New(func(name string, payload any) {
		if name == "plan" {
			close(cardStarted)
			<-releaseCard
		}
		events <- event{name, payload}
	})
	j := job("message", "conversation")
	done := make(chan cogito.PlanDecision, 1)
	go func() {
		done <- r.ApprovePlan(j, j.GetContext(), &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
	}()
	awaitEvent(t, events, "json_message_status")
	<-cardStarted
	pending := r.Pending("conversation").Plan
	if pending == nil {
		t.Fatal("plan is not reservable while its event is being delivered")
	}
	if err := r.Decide(pending.ID, true, nil, ""); err != nil {
		t.Fatal(err)
	}
	select {
	case got := <-events:
		t.Fatalf("event arrived before plan callback returned: %#v", got)
	default:
	}
	select {
	case got := <-done:
		t.Fatalf("approval resumed before plan callback returned: %#v", got)
	default:
	}
	close(releaseCard)
	awaitEvent(t, events, "plan")
	awaitEvent(t, events, "json_message_status")
	if got := <-done; !got.Approved || got.Plan != nil {
		t.Fatalf("decision = %#v", got)
	}
}

func TestPreCancelledPlanIsNotPublished(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var events []event
	r := New(func(name string, payload any) { events = append(events, event{name, payload}) })
	decision := r.ApprovePlan(job("message", "conversation"), ctx, &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
	if decision.Approved || decision.Plan != nil || decision.Feedback != "" {
		t.Fatalf("decision = %#v", decision)
	}
	if len(events) != 0 {
		t.Fatalf("pre-cancelled plan emitted events: %#v", events)
	}
	if got := r.Pending("conversation"); len(got.Questions) != 0 || got.Plan != nil {
		t.Fatalf("pre-cancelled plan is pending: %#v", got)
	}
}

func TestCancelAllCancelsBlockedJobsAndPlansRejectOnContext(t *testing.T) {
	events := make(chan event, 8)
	r := New(func(name string, payload any) { events <- event{name, payload} })
	questionJob := job("question-message", "conversation")
	planJob := job("plan-message", "conversation")
	questionDone := make(chan error, 1)
	planDone := make(chan cogito.PlanDecision, 1)
	go func() {
		_, err := r.HandleQuestion(questionJob, questionJob.GetContext(), question("q", "Tell", nil, true, time.Now()))
		questionDone <- err
	}()
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "question")
	go func() {
		planDone <- r.ApprovePlan(planJob, planJob.GetContext(), &structures.Plan{Description: "proposal", Subtasks: []string{"one"}}, nil)
	}()
	awaitEvent(t, events, "json_message_status")
	awaitEvent(t, events, "plan")

	r.CancelAll()
	if err := <-questionDone; !errors.Is(err, cogito.ErrQuestionCancelled) {
		t.Fatalf("question error = %v", err)
	}
	if got := <-planDone; got.Approved || got.Plan != nil || got.Feedback != "" {
		t.Fatalf("cancelled plan decision = %#v", got)
	}
	if got := r.Pending("conversation"); len(got.Questions) != 0 || got.Plan != nil {
		t.Fatalf("pending after CancelAll = %#v", got)
	}
}
