package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

// ErrConversationBusy means the live loop cannot accept another queued message.
// Callers should retry instead of starting a replacement job.
var ErrConversationBusy = errors.New("conversation input queue is full")

// WithSubAgents enables Cogito's delegation tools and parked conversation loop.
func WithSubAgents(enabled bool) Option {
	return func(o *options) error { o.enableSubAgents = enabled; return nil }
}

// WithSubAgentProvider resolves allowed definitions and a dispatcher per job.
func WithSubAgentProvider(provider func(*types.Job) ([]cogito.AgentDefinition, cogito.AgentDispatcher)) Option {
	return func(o *options) error { o.subAgentProvider = provider; return nil }
}

type conversationLoop struct {
	mu                          sync.Mutex
	job                         *types.Job
	inbox                       chan openai.ChatCompletionMessage
	done                        chan struct{}
	accepting, parked, terminal bool
	reserved                    int
	settled                     *sync.Cond
	replies                     uint64
	children                    map[string]context.CancelFunc
}

// InjectChat sends a message into a conversation which has parked at least once.
// It also accepts messages while that loop is waking or processing a completion.
// A false result means the caller should start a normal history-backed job.
func (a *Agent) InjectChat(conversationID, message, messageID string) (bool, error) {
	if conversationID == "" {
		return false, nil
	}
	a.liveMu.Lock()
	loop := a.liveConversations[conversationID]
	a.liveMu.Unlock()
	if loop == nil {
		return false, nil
	}
	loop.mu.Lock()
	if !loop.accepting || loop.job.GetContext().Err() != nil {
		wait := loop.parked && (loop.terminal || loop.job.GetContext().Err() != nil)
		loop.mu.Unlock()
		if wait {
			<-loop.done
		}
		return false, nil
	}
	if len(loop.inbox)+loop.reserved >= cap(loop.inbox) {
		loop.mu.Unlock()
		return true, ErrConversationBusy
	}
	loop.reserved++
	loop.mu.Unlock()
	a.emitJobEvent(loop.job, "json_message", map[string]any{"id": messageID + "-user", "message_id": messageID, "sender": "user", "content": message})
	loop.mu.Lock()
	defer loop.mu.Unlock()
	loop.reserved--
	if loop.settled != nil {
		loop.settled.Broadcast()
	}
	if err := loop.job.GetContext().Err(); err != nil {
		return true, err
	}
	select {
	case loop.inbox <- openai.ChatCompletionMessage{Role: UserRole, Content: message}:
		return true, nil
	default:
		return true, ErrConversationBusy
	}
}

// pending is called only after Cogito's manager has no running children or
// unpublished completions. Admission and terminal detection share one lock.
func (l *conversationLoop) pending() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.inbox) > 0 || l.reserved > 0 {
		return true
	}
	l.accepting = false
	l.terminal = true
	return false
}

// seal closes admission and waits for messages already reserved by callers.
// The caller can then drain the inbox without racing an accepted user message.
func (l *conversationLoop) seal() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.accepting = false
	l.terminal = true
	for l.reserved > 0 {
		l.settled.Wait()
	}
}

func (l *conversationLoop) finish(fragment cogito.Fragment) cogito.Fragment {
	l.seal()
	for {
		select {
		case message := <-l.inbox:
			fragment.Messages = append(fragment.Messages, message)
		default:
			return fragment
		}
	}
}

func (a *Agent) delegationOptions(job *types.Job) ([]cogito.Option, func(), func(cogito.Fragment) cogito.Fragment) {
	loop := &conversationLoop{job: job, inbox: make(chan openai.ChatCompletionMessage, 256), done: make(chan struct{}), children: make(map[string]context.CancelFunc)}
	loop.settled = sync.NewCond(&loop.mu)
	conversationID, _ := job.Metadata[types.MetadataKeyConversationID].(string)
	// Delegated jobs retain parent identity for events, but never own the parent's
	// interactive conversation on a peer that may run several children at once.
	delegationID, _ := job.Metadata[types.MetadataKeyDelegationID].(string)
	if conversationID != "" && delegationID == "" {
		a.liveMu.Lock()
		a.liveConversations[conversationID] = loop
		a.liveMu.Unlock()
	}
	cleanup := func() {
		loop.mu.Lock()
		loop.accepting = false
		loop.terminal = true
		cancels := make([]context.CancelFunc, 0, len(loop.children))
		for _, cancel := range loop.children {
			cancels = append(cancels, cancel)
		}
		loop.mu.Unlock()
		for _, cancel := range cancels {
			cancel()
		}
		a.liveMu.Lock()
		if a.liveConversations[conversationID] == loop {
			delete(a.liveConversations, conversationID)
		}
		a.liveMu.Unlock()
		close(loop.done)
	}
	opts := []cogito.Option{
		cogito.EnableAgentSpawning,
		cogito.WithMessageInjectionChan(loop.inbox),
		cogito.WithPendingWork(loop.pending),
		cogito.WithOnBeforeFinalResponse(loop.seal),
		cogito.WithOnParkSnapshot(func(fragment cogito.Fragment, reply string) {
			loop.mu.Lock()
			loop.parked = true
			loop.accepting = true
			loop.replies++
			number := loop.replies
			loop.mu.Unlock()
			if reply == "" {
				reply = "Work is continuing in the background."
				fragment.Messages = append(fragment.Messages, openai.ChatCompletionMessage{Role: AssistantRole, Content: reply})
			}
			if conversationID != "" && delegationID == "" {
				a.sharedState.ConversationTracker.SetConversation(conversationID, fragment.Messages)
			}
			a.emitJobEvent(job, "json_message", map[string]any{"id": fmt.Sprintf("%s-agent-park-%d", job.UUID, number), "sender": "agent", "content": reply})
			a.emitJobEvent(job, "json_message_status", map[string]any{"status": "waiting_agents"})
		}),
		cogito.WithOnResume(func() { a.emitJobEvent(job, "json_message_status", map[string]any{"status": "processing"}) }),
		cogito.WithAgentSpawnCallback(func(child *cogito.AgentState) {
			loop.mu.Lock()
			loop.children[child.ID] = child.Cancel
			terminal := loop.terminal
			loop.mu.Unlock()
			if terminal || job.GetContext().Err() != nil {
				child.Cancel()
				return
			}
			a.emitSubAgent(job, child, "spawned")
		}),
		cogito.WithAgentCompletionCallback(func(child *cogito.AgentState) {
			loop.mu.Lock()
			delete(loop.children, child.ID)
			loop.mu.Unlock()
			if job.GetContext().Err() != nil {
				return
			}
			status := "completed"
			if child.Error != nil {
				status = "failed"
			}
			a.emitSubAgent(job, child, status)
		}),
	}
	if a.options.subAgentProvider != nil {
		definitions, dispatcher := a.options.subAgentProvider(job)
		opts = append(opts, cogito.WithAgentDefinitions(definitions...), cogito.WithAgentDispatcher(dispatcher))
	}
	return opts, cleanup, loop.finish
}

func (a *Agent) emitSubAgent(job *types.Job, child *cogito.AgentState, status string) {
	summary := child.Result
	if len(summary) > 1024 {
		summary = strings.ToValidUTF8(summary[:1024], "")
	}
	a.emitJobEvent(job, "sub_agent", map[string]any{"agent_id": child.ID, "agent_type": child.Type, "status": status, "background": child.Background, "task": child.Task, "result_summary": summary})
}

func (a *Agent) emitJobEvent(job *types.Job, event string, payload map[string]any) {
	if id, ok := job.Metadata[types.MetadataKeyConversationID].(string); ok && id != "" {
		payload["conversation_id"] = id
	}
	if _, ok := payload["message_id"]; !ok {
		payload["message_id"] = job.UUID
	}
	payload["timestamp"] = time.Now().Format(time.RFC3339)
	if job.EventCallback != nil {
		job.EventCallback(event, payload)
	} else if a.options.interactionCallback != nil {
		a.options.interactionCallback(event, payload)
	}
}

func (a *Agent) saveLiveConversation(job *types.Job) {
	if !a.options.enableSubAgents {
		return
	}
	if id, _ := job.Metadata[types.MetadataKeyDelegationID].(string); id != "" {
		return
	}
	if id, _ := job.Metadata[types.MetadataKeyConversationID].(string); id != "" {
		a.sharedState.ConversationTracker.SetConversation(id, job.Result.Conversation)
		job.Result.ConversationSaved = true
	}
}

func jobCancellationKey(job *types.Job) string {
	if id, _ := job.Metadata[types.MetadataKeyDelegationID].(string); id != "" {
		return ""
	}
	id, _ := job.Metadata[types.MetadataKeyConversationID].(string)
	return id
}
