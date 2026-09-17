package state

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	coreagent "github.com/mudler/LocalAGI/core/agent"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/LocalAGI/pkg/config"
	"github.com/mudler/cogito"
)

func TestAgentConfigSubAgentJSONAndMetadata(t *testing.T) {
	var cfg AgentConfig
	data := `{"enable_sub_agents":true,"sub_agents":["researcher"],"remote_agents":[{"name":"remote","description":"Remote worker","url":"https://example.test","api_key":"secret"}]}`
	if err := json.Unmarshal([]byte(data), &cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.EnableSubAgents || len(cfg.SubAgents) != 1 || cfg.SubAgents[0] != "researcher" {
		t.Fatalf("unexpected sub-agent config: %+v", cfg)
	}
	if len(cfg.RemoteAgents) != 1 || cfg.RemoteAgents[0].APIKey != "secret" {
		t.Fatalf("unexpected remote-agent config: %+v", cfg.RemoteAgents)
	}

	meta := NewAgentConfigMeta(nil, nil, nil, nil)
	fields := make(map[string]config.Field, len(meta.Fields))
	for _, field := range meta.Fields {
		fields[field.Name] = field
	}
	if field, ok := fields["enable_sub_agents"]; !ok || field.DefaultValue != false || field.Type != config.FieldTypeCheckbox {
		t.Fatalf("enable_sub_agents metadata missing or invalid: %+v", field)
	}
	for _, name := range []string{"sub_agents", "remote_agents"} {
		if _, ok := fields[name]; ok {
			t.Fatalf("%s must not use the generic text editor for structured config", name)
		}
	}
}

func TestSubAgentProviderBuildsCurrentAllowlistedDefinitionsAndRemoteWins(t *testing.T) {
	pool := &AgentPool{pool: AgentPoolData{
		"parent": {Name: "parent"},
		"local":  {Name: "local", Description: "local description", SystemPrompt: "local prompt", Model: "local model"},
		"other":  {Name: "other", Description: "other description"},
	}}
	cfg := &AgentConfig{
		EnableSubAgents: true,
		SubAgents:       []string{"local"},
		RemoteAgents: []RemoteAgent{{
			Name: "local", Description: "remote description", URL: "http://remote.invalid", APIKey: "must-not-be-metadata",
		}},
	}

	provider := pool.subAgentProvider("parent", cfg)
	pool.Lock()
	pool.pool["late"] = AgentConfig{Name: "late", Description: "loaded later"}
	pool.Unlock()
	defs, _ := provider(types.NewJob())

	if len(defs) != 1 {
		t.Fatalf("got %d definitions, want 1: %+v", len(defs), defs)
	}
	got := defs[0]
	if got.Name != "local" || got.Description != "remote description" {
		t.Fatalf("remote definition did not win collision: %+v", got)
	}
	if got.SystemPrompt != "" || got.Model != "" || len(got.Metadata) != 0 {
		t.Fatalf("remote definition exposed local or credential details: %+v", got)
	}

	cfg.SubAgents = nil
	cfg.RemoteAgents = nil
	defs, _ = pool.subAgentProvider("parent", cfg)(types.NewJob())
	want := map[string]bool{"late": true, "local": true, "other": true}
	if len(defs) != len(want) {
		t.Fatalf("got definitions %+v, want all peers", defs)
	}
	for _, def := range defs {
		if !want[def.Name] {
			t.Fatalf("unexpected definition %+v", def)
		}
		delete(want, def.Name)
	}
}

func TestSubAgentDispatcherFallbackAndMissingAllowedPeer(t *testing.T) {
	pool := &AgentPool{pool: AgentPoolData{"parent": {Name: "parent"}}}
	job := types.NewJob()

	_, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true})(job)
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "unknown"}); !errors.Is(err, cogito.ErrDispatchFallback) {
		t.Fatalf("unknown type error = %v, want ErrDispatchFallback", err)
	}

	_, dispatch = pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true, SubAgents: []string{"missing"}})(job)
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "missing"}); err == nil || errors.Is(err, cogito.ErrDispatchFallback) {
		t.Fatalf("explicitly allowed missing peer error = %v, want a non-fallback error", err)
	}
}

type captureDelegatedJobFilter struct {
	mu  sync.Mutex
	job *types.Job
}

func (f *captureDelegatedJobFilter) Name() string    { return "capture delegated job" }
func (f *captureDelegatedJobFilter) IsTrigger() bool { return false }
func (f *captureDelegatedJobFilter) Apply(job *types.Job) (bool, error) {
	f.mu.Lock()
	f.job = job
	f.mu.Unlock()
	job.Result.SetResponse("local answer")
	return false, nil
}

func TestSubAgentDispatcherInvokesPoolAgentWithDelegationMetadata(t *testing.T) {
	filter := &captureDelegatedJobFilter{}
	peer, err := coreagent.New(
		coreagent.WithCharacter(coreagent.Character{Name: "peer"}),
		coreagent.WithJobFilters(filter),
		coreagent.WithSchedulerStorePath(filepath.Join(t.TempDir(), "scheduled_tasks.json")),
	)
	if err != nil {
		t.Fatal(err)
	}
	go peer.Run()
	t.Cleanup(peer.Stop)

	pool := &AgentPool{
		pool:   AgentPoolData{"parent": {Name: "parent"}, "peer": {Name: "peer", Description: "helper"}},
		agents: map[string]*coreagent.Agent{"peer": peer},
	}
	parent := types.NewJob(
		types.WithUUID("parent-message"),
		types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation-1"}),
	)
	_, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true})(parent)
	fragment, err := dispatch(context.Background(), cogito.AgentRunSpec{ID: "delegation-1", Type: "peer", Task: "do work"})
	if err != nil {
		t.Fatal(err)
	}
	if len(fragment.Messages) != 1 || fragment.Messages[0].Content != "local answer" {
		t.Fatalf("unexpected fragment: %+v", fragment.Messages)
	}

	filter.mu.Lock()
	delegated := filter.job
	filter.mu.Unlock()
	if delegated == nil || len(delegated.ConversationHistory) != 1 || delegated.ConversationHistory[0].Content != "do work" {
		t.Fatalf("delegated job did not carry task: %+v", delegated)
	}
	wantMetadata := map[string]any{
		types.MetadataKeyConversationID: "conversation-1",
		"parent_agent_id":               "parent",
		"parent_message_id":             "parent-message",
		types.MetadataKeyDelegationID:   "delegation-1",
	}
	for key, want := range wantMetadata {
		if got := delegated.Metadata[key]; got != want {
			t.Errorf("metadata %q = %#v, want %#v", key, got, want)
		}
	}
}

func TestRemoteSubAgentDispatchPostsResponsesRequestWithAuth(t *testing.T) {
	var gotPath, gotAuth, gotBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"status":"completed","output":[{"content":[{"type":"output_text","text":"remote "},{"type":"output_text","text":"answer"}]}]}`)
	}))
	defer server.Close()

	pool := &AgentPool{pool: AgentPoolData{"parent": {Name: "parent"}}}
	cfg := &AgentConfig{EnableSubAgents: true, RemoteAgents: []RemoteAgent{{Name: "remote", URL: server.URL + "/", APIKey: "top-secret"}}}
	_, dispatch := pool.subAgentProvider("parent", cfg)(types.NewJob())
	fragment, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote", Task: "investigate"})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/v1/responses" || gotAuth != "Bearer top-secret" {
		t.Fatalf("path/auth = %q/%q", gotPath, gotAuth)
	}
	if !strings.Contains(gotBody, `"model":"remote"`) || !strings.Contains(gotBody, `"input":"investigate"`) {
		t.Fatalf("unexpected request body: %s", gotBody)
	}
	if len(fragment.Messages) != 1 || fragment.Messages[0].Content != "remote answer" {
		t.Fatalf("unexpected remote fragment: %+v", fragment.Messages)
	}
}

func TestRemoteSubAgentDispatchErrorsAreBoundedAndRedacted(t *testing.T) {
	const secret = "api-secret-value"
	t.Run("non-2xx", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, "upstream dumped "+secret)
		}))
		defer server.Close()
		dispatch := remoteTestDispatcher(t, server.URL, secret)
		_, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote", Task: "task"})
		if err == nil || strings.Contains(err.Error(), secret) || !strings.Contains(err.Error(), "502") {
			t.Fatalf("unexpected non-2xx error: %v", err)
		}
	})

	t.Run("transport", func(t *testing.T) {
		remoteURL := "http://127.0.0.1:1/private-path"
		dispatch := remoteTestDispatcher(t, remoteURL, secret)
		_, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote", Task: "task"})
		if err == nil || strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), remoteURL) {
			t.Fatalf("transport error leaked remote details: %v", err)
		}
	})

	t.Run("response failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			io.WriteString(w, `{"status":"failed","error":{"code":"worker_failed","message":"could not finish"}}`)
		}))
		defer server.Close()
		dispatch := remoteTestDispatcher(t, server.URL, secret)
		_, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote", Task: "task"})
		if err == nil || !strings.Contains(err.Error(), "worker_failed") || !strings.Contains(err.Error(), "could not finish") {
			t.Fatalf("unexpected response failure: %v", err)
		}
	})
}

func TestRemoteSubAgentDispatchUsesContextCancellation(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer server.Close()
	defer close(release)

	dispatch := remoteTestDispatcher(t, server.URL, "")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := dispatch(ctx, cogito.AgentRunSpec{Type: "remote", Task: "task"})
		done <- err
	}()
	<-started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("remote dispatch did not honor context cancellation")
	}
}

func remoteTestDispatcher(t *testing.T, url, key string) cogito.AgentDispatcher {
	t.Helper()
	pool := &AgentPool{pool: AgentPoolData{"parent": {Name: "parent"}}}
	_, dispatch := pool.subAgentProvider("parent", &AgentConfig{
		EnableSubAgents: true,
		RemoteAgents:    []RemoteAgent{{Name: "remote", URL: url, APIKey: key}},
	})(types.NewJob())
	return dispatch
}
