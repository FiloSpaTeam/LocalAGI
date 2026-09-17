// Package interactions coordinates questions and plan approvals that pause an
// agent job while a user supplies input.
package interactions

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
)

var (
	ErrPlanNotFound        = errors.New("interaction plan not found or already decided")
	ErrInvalidPlanDecision = errors.New("invalid interaction plan decision")
	ErrFreeTextNotAllowed  = errors.New("free text is not allowed for this question")
)

// Question is a pending Cogito question with the job identity needed by a UI.
type Question struct {
	cogito.UserQuestion
	ConversationID string    `json:"conversation_id,omitempty"`
	MessageID      string    `json:"message_id"`
	Timestamp      time.Time `json:"timestamp"`
}

// Plan is a pending plan approval with the job identity needed by a UI.
type Plan struct {
	AgentID        string    `json:"agent_id,omitempty"`
	ID             string    `json:"id"`
	ConversationID string    `json:"conversation_id,omitempty"`
	MessageID      string    `json:"message_id"`
	Description    string    `json:"description"`
	Subtasks       []string  `json:"subtasks"`
	Timestamp      time.Time `json:"timestamp"`
}

// Snapshot contains pending interactions for one exact conversation identity.
type Snapshot struct {
	Questions []Question `json:"questions"`
	Plan      *Plan      `json:"plan"`
}

type questionRecord struct {
	question  Question
	job       *types.Job
	ctx       context.Context
	cancel    context.CancelFunc
	published chan struct{}
	sequence  uint64
	active    bool
}

type planRecord struct {
	plan      Plan
	original  structures.Plan
	job       *types.Job
	ctx       context.Context
	cancel    context.CancelFunc
	decision  chan cogito.PlanDecision
	published chan struct{}
	sequence  uint64
	active    bool
}

// Registry owns the single Cogito question registry shared by one LocalAGI
// agent, plus its pending plan approval.
type Registry struct {
	mu        sync.Mutex
	emit      func(event string, payload any)
	questions *cogito.QuestionRegistry
	question  map[string]*questionRecord
	plans     map[string]*planRecord
	sequence  uint64
}

// New creates an interaction registry. emit may be nil.
func New(emit func(event string, payload any)) *Registry {
	r := &Registry{
		emit:     emit,
		question: make(map[string]*questionRecord),
		plans:    make(map[string]*planRecord),
	}
	r.questions = cogito.NewQuestionRegistry(r.onQuestion)
	return r
}

// HandleQuestion records the job identity around Cogito's blocking question
// registry and removes the index when the answer or context completes.
func (r *Registry) HandleQuestion(job *types.Job, ctx context.Context, q cogito.UserQuestion) (cogito.UserAnswer, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	wrapped, cancel := context.WithCancel(ctx)
	if q.AgentID == "" {
		q.AgentID = delegationID(job)
	}
	record := &questionRecord{
		question: Question{
			UserQuestion:   cloneUserQuestion(q),
			ConversationID: conversationID(job),
			MessageID:      messageID(job),
			Timestamp:      q.AskedAt,
		},
		job:       job,
		ctx:       wrapped,
		cancel:    cancel,
		published: make(chan struct{}),
	}

	r.mu.Lock()
	if _, exists := r.question[q.ID]; !exists {
		r.sequence++
		record.sequence = r.sequence
		r.question[q.ID] = record
	}
	r.mu.Unlock()
	defer func() {
		cancel()
		r.mu.Lock()
		if r.question[q.ID] == record {
			delete(r.question, q.ID)
		}
		r.mu.Unlock()
	}()

	return r.questions.Handle(wrapped, q)
}

func (r *Registry) onQuestion(q cogito.UserQuestion) {
	r.mu.Lock()
	record := r.question[q.ID]
	if record == nil {
		r.mu.Unlock()
		return
	}
	defer close(record.published)
	if record.ctx.Err() != nil {
		r.mu.Unlock()
		return
	}
	shown := cloneQuestion(record.question)
	r.mu.Unlock()

	r.emitJobEvent(record.job, "json_message_status", statusPayload("waiting_user", shown.ConversationID, shown.MessageID))

	r.mu.Lock()
	stillPending := r.question[q.ID] == record && record.ctx.Err() == nil
	if stillPending {
		record.active = true
	}
	r.mu.Unlock()
	if stillPending {
		r.emitJobEvent(record.job, "question", shown)
	}
}

func (r *Registry) deliverAnswer(record *questionRecord, id string, answer cogito.UserAnswer) error {
	if record.ctx.Err() != nil {
		return cogito.ErrQuestionNotFound
	}
	r.emitJobEvent(record.job, "json_message_status", statusPayload("processing", record.question.ConversationID, record.question.MessageID))
	if record.ctx.Err() != nil {
		return cogito.ErrQuestionNotFound
	}
	return r.questions.Answer(id, answer)
}

// Answer validates and releases one pending question.
func (r *Registry) Answer(id string, answer cogito.UserAnswer) error {
	r.mu.Lock()
	record := r.question[id]
	if record == nil || !record.active || record.ctx.Err() != nil {
		r.mu.Unlock()
		return cogito.ErrQuestionNotFound
	}
	if err := record.question.UserQuestion.Validate(answer); err != nil {
		r.mu.Unlock()
		return err
	}
	record.active = false
	answer.Selected = slices.Clone(answer.Selected)
	r.mu.Unlock()

	select {
	case <-record.published:
		return r.deliverAnswer(record, id, answer)
	default:
		go func() {
			select {
			case <-record.published:
				_ = r.deliverAnswer(record, id, answer)
			case <-record.ctx.Done():
			}
		}()
		return nil
	}
}

// ApprovePlan blocks until the user decides, or until its context is canceled.
func (r *Registry) ApprovePlan(job *types.Job, ctx context.Context, plan *structures.Plan, _ *structures.Goal) cogito.PlanDecision {
	if ctx == nil {
		ctx = context.Background()
	}
	if ctx.Err() != nil {
		return cogito.PlanDecision{}
	}
	wrapped, cancel := context.WithCancel(ctx)
	original := clonePlan(plan)
	record := &planRecord{
		plan: Plan{
			AgentID:        delegationID(job),
			ID:             uuid.NewString(),
			ConversationID: conversationID(job),
			MessageID:      messageID(job),
			Description:    original.Description,
			Subtasks:       slices.Clone(original.Subtasks),
			Timestamp:      time.Now(),
		},
		original:  original,
		job:       job,
		ctx:       wrapped,
		cancel:    cancel,
		decision:  make(chan cogito.PlanDecision, 1),
		published: make(chan struct{}),
	}
	r.mu.Lock()
	r.sequence++
	record.sequence = r.sequence
	r.plans[record.plan.ID] = record
	r.mu.Unlock()
	defer func() {
		cancel()
		r.mu.Lock()
		if r.plans[record.plan.ID] == record {
			delete(r.plans, record.plan.ID)
		}
		r.mu.Unlock()
	}()

	r.publishPlan(record)

	select {
	case decision := <-record.decision:
		return decision
	case <-wrapped.Done():
		return cogito.PlanDecision{}
	}
}

func (r *Registry) publishPlan(record *planRecord) {
	defer close(record.published)
	if record.ctx.Err() != nil {
		return
	}
	r.emitJobEvent(record.job, "json_message_status", statusPayload("waiting_user", record.plan.ConversationID, record.plan.MessageID))
	r.mu.Lock()
	stillPending := r.plans[record.plan.ID] == record && record.ctx.Err() == nil
	if stillPending {
		record.active = true
	}
	r.mu.Unlock()
	if stillPending {
		r.emitJobEvent(record.job, "plan", clonePublicPlan(record.plan))
	}
}

// Decide releases one pending plan approval.
func (r *Registry) Decide(id string, approved bool, subtasks *[]string, feedback string) error {
	r.mu.Lock()
	record := r.plans[id]
	if record == nil || !record.active || record.ctx.Err() != nil {
		r.mu.Unlock()
		return ErrPlanNotFound
	}
	decision := cogito.PlanDecision{Approved: approved, Feedback: feedback}
	if approved && subtasks != nil {
		if len(*subtasks) == 0 {
			r.mu.Unlock()
			return fmt.Errorf("%w: an approved edited plan needs at least one subtask", ErrInvalidPlanDecision)
		}
		for _, subtask := range *subtasks {
			if strings.TrimSpace(subtask) == "" {
				r.mu.Unlock()
				return fmt.Errorf("%w: edited subtasks must not be blank", ErrInvalidPlanDecision)
			}
		}
		decision.Plan = &structures.Plan{Description: record.original.Description, Subtasks: slices.Clone(*subtasks)}
	}
	record.active = false
	r.mu.Unlock()

	select {
	case <-record.published:
		return r.deliverDecision(record, decision)
	default:
		go func() {
			select {
			case <-record.published:
				_ = r.deliverDecision(record, decision)
			case <-record.ctx.Done():
			}
		}()
		return nil
	}
}

func (r *Registry) deliverDecision(record *planRecord, decision cogito.PlanDecision) error {
	if record.ctx.Err() != nil {
		return ErrPlanNotFound
	}
	r.emitJobEvent(record.job, "json_message_status", statusPayload("processing", record.plan.ConversationID, record.plan.MessageID))
	if record.ctx.Err() != nil {
		return ErrPlanNotFound
	}
	record.decision <- decision
	return nil
}

// AnswerText routes free text to the oldest pending question in the exact
// non-empty conversation.
func (r *Registry) AnswerText(conversationID, text string) (questionID string, handled bool, err error) {
	if conversationID == "" {
		return "", false, nil
	}
	r.mu.Lock()
	var oldest *questionRecord
	for _, record := range r.question {
		if !record.active || record.ctx.Err() != nil || record.question.ConversationID != conversationID {
			continue
		}
		if oldest == nil || questionLess(record, oldest) {
			oldest = record
		}
	}
	if oldest == nil {
		r.mu.Unlock()
		return "", false, nil
	}
	id := oldest.question.ID
	allowed := oldest.question.AllowFreeText
	r.mu.Unlock()
	if !allowed {
		return id, true, ErrFreeTextNotAllowed
	}
	return id, true, r.Answer(id, cogito.UserAnswer{Text: text})
}

// Pending returns owned copies of interactions for exactly conversationID.
func (r *Registry) Pending(conversationID string) Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()

	questions := make([]*questionRecord, 0)
	for _, record := range r.question {
		if record.active && record.ctx.Err() == nil && record.question.ConversationID == conversationID {
			questions = append(questions, record)
		}
	}
	sort.Slice(questions, func(i, j int) bool { return questionLess(questions[i], questions[j]) })
	snapshot := Snapshot{Questions: make([]Question, 0, len(questions))}
	for _, record := range questions {
		snapshot.Questions = append(snapshot.Questions, cloneQuestion(record.question))
	}

	var oldest *planRecord
	for _, record := range r.plans {
		if !record.active || record.ctx.Err() != nil || record.plan.ConversationID != conversationID {
			continue
		}
		if oldest == nil || record.plan.Timestamp.Before(oldest.plan.Timestamp) ||
			(record.plan.Timestamp.Equal(oldest.plan.Timestamp) && record.sequence < oldest.sequence) {
			oldest = record
		}
	}
	if oldest != nil {
		plan := clonePublicPlan(oldest.plan)
		snapshot.Plan = &plan
	}
	return snapshot
}

// CancelAll cancels every job currently blocked on an interaction.
func (r *Registry) CancelAll() {
	r.mu.Lock()
	jobs := make(map[*types.Job]struct{})
	cancels := make([]context.CancelFunc, 0, len(r.question)+len(r.plans))
	for id, record := range r.question {
		record.active = false
		delete(r.question, id)
		cancels = append(cancels, record.cancel)
		if record.job != nil {
			jobs[record.job] = struct{}{}
		}
	}
	for id, record := range r.plans {
		record.active = false
		delete(r.plans, id)
		cancels = append(cancels, record.cancel)
		if record.job != nil {
			jobs[record.job] = struct{}{}
		}
	}
	r.mu.Unlock()

	for _, cancel := range cancels {
		cancel()
	}
	for job := range jobs {
		job.Cancel()
	}
}

func (r *Registry) emitJobEvent(job *types.Job, event string, payload any) {
	if event == "json_message_status" {
		if status, ok := payload.(map[string]any); ok {
			if id := delegationID(job); id != "" {
				status["agent_id"] = id
			}
			if status["status"] == "processing" {
				pending := r.Pending(conversationID(job))
				if len(pending.Questions) > 0 || pending.Plan != nil {
					status["status"] = "waiting_user"
				}
			}
		}
	}

	if job != nil && job.EventCallback != nil {
		job.EventCallback(event, payload)
		return
	}
	if r.emit != nil {
		r.emit(event, payload)
	}
}

func statusPayload(status, conversationID, messageID string) map[string]any {
	payload := map[string]any{
		"status":     status,
		"message_id": messageID,
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
	}
	if conversationID != "" {
		payload["conversation_id"] = conversationID
	}
	return payload
}

func conversationID(job *types.Job) string {
	if job == nil || job.Metadata == nil {
		return ""
	}
	id, _ := job.Metadata[types.MetadataKeyConversationID].(string)
	return id
}

func messageID(job *types.Job) string {
	if job == nil {
		return ""
	}
	if delegationID(job) != "" {
		if id, _ := job.Metadata[types.MetadataKeyParentMessageID].(string); id != "" {
			return id
		}
	}
	return job.UUID
}

func delegationID(job *types.Job) string {
	if job == nil {
		return ""
	}
	id, _ := job.Metadata[types.MetadataKeyDelegationID].(string)
	return id
}

func questionLess(a, b *questionRecord) bool {
	if a.question.AskedAt.Equal(b.question.AskedAt) {
		return a.sequence < b.sequence
	}
	return a.question.AskedAt.Before(b.question.AskedAt)
}

func cloneUserQuestion(q cogito.UserQuestion) cogito.UserQuestion {
	q.Options = slices.Clone(q.Options)
	return q
}

func cloneQuestion(q Question) Question {
	q.UserQuestion = cloneUserQuestion(q.UserQuestion)
	return q
}

func clonePlan(plan *structures.Plan) structures.Plan {
	if plan == nil {
		return structures.Plan{}
	}
	return structures.Plan{Description: plan.Description, Subtasks: slices.Clone(plan.Subtasks)}
}

func clonePublicPlan(plan Plan) Plan {
	plan.Subtasks = slices.Clone(plan.Subtasks)
	return plan
}
