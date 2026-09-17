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
	parentEvents := 0
	parent := types.NewJob(
		types.WithInteractionHandler(peer.Interactions()),
		types.WithEventCallback(func(string, any) { parentEvents++ }),
		types.WithUUID("parent-message"),
		types.WithMetadata(map[string]any{types.MetadataKeyConversationID: "conversation-1", "tenant": "tenant-a", "parent_message_id": "root-message"}),
	)
	pool.SetSubAgentResolver(func(_ string, _ *types.Job, candidate string) (string, bool) {
		return "visible-" + candidate, true
	})
	_, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true, SubAgents: []string{"visible-peer"}})(parent)
	fragment, err := dispatch(context.Background(), cogito.AgentRunSpec{ID: "delegation-1", Type: "visible-peer", Task: "do work"})
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
	if delegated.InteractionHandler != parent.InteractionHandler {
		t.Fatal("dispatcher lost root interaction handler")
	}
	if delegated.EventCallback == nil {
		t.Fatal("dispatcher lost root event sink")
	}
	delegated.EventCallback("test", nil)
	if parentEvents != 1 {
		t.Fatal("dispatcher replaced root event sink")
	}
	wantMetadata := map[string]any{
		types.MetadataKeyConversationID: "conversation-1",
		"parent_agent_id":               "parent",
		"parent_message_id":             "root-message",
		"tenant":                        "tenant-a",
		types.MetadataKeyDelegationID:   "delegation-1",
	}
	if _, exists := parent.Metadata["parent_agent_id"]; exists {
		t.Fatal("dispatch mutated parent metadata")
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

func TestSubAgentResolverScopesAliasesAndRevocation(t *testing.T) {
	pool := &AgentPool{pool: AgentPoolData{
		"tenant-a/parent": {Name: "parent"}, "tenant-a/worker": {Name: "worker", SystemPrompt: "authorized prompt"},
		"tenant-b/worker": {Name: "worker", SystemPrompt: "private prompt"},
		"tenant-a/second": {Name: "second", SystemPrompt: "unlisted prompt"},
	}}
	job := types.NewJob(types.WithMetadata(map[string]any{"tenant": "tenant-a"}))
	pool.SetSubAgentResolver(func(parent string, gotJob *types.Job, candidate string) (string, bool) {
		if parent != "tenant-a/parent" || gotJob != job {
			t.Errorf("resolver lost caller context")
		}
		return strings.TrimPrefix(candidate, "tenant-a/"), strings.HasPrefix(candidate, "tenant-a/")
	})
	defs, dispatch := pool.subAgentProvider("tenant-a/parent", &AgentConfig{EnableSubAgents: true, SubAgents: []string{"worker"}})(job)
	if len(defs) != 1 || defs[0].Name != "worker" || defs[0].SystemPrompt != "authorized prompt" {
		t.Fatalf("scoped definitions = %+v", defs)
	}
	for _, name := range []string{"tenant-a/worker", "tenant-b/worker", "second", "parent", "unknown"} {
		if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: name}); err == nil || errors.Is(err, cogito.ErrDispatchFallback) {
			t.Errorf("%s did not fail closed: %v", name, err)
		}
	}
	pool.SetSubAgentResolver(func(string, *types.Job, string) (string, bool) { return "", false })
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "worker"}); err == nil || !strings.Contains(err.Error(), "authorized") {
		t.Fatalf("revoked dispatch = %v", err)
	}
}

func TestSubAgentResolverRejectsAmbiguousAliases(t *testing.T) {
	pool := &AgentPool{pool: AgentPoolData{"parent": {}, "a": {}, "b": {}}}
	pool.SetSubAgentResolver(func(_ string, _ *types.Job, candidate string) (string, bool) {
		if candidate == "parent" {
			return "self", true
		}
		return "same", true
	})
	defs, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true})(types.NewJob())
	if len(defs) != 0 {
		t.Fatalf("ambiguous definitions = %+v", defs)
	}
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "same"}); err == nil || errors.Is(err, cogito.ErrDispatchFallback) {
		t.Fatalf("ambiguous dispatch = %v", err)
	}
}

type delegationHTTPDoer func(*http.Request) (*http.Response, error)

func (f delegationHTTPDoer) Do(r *http.Request) (*http.Response, error) { return f(r) }

func TestRemoteSubAgentUsesSuppliedHTTPClient(t *testing.T) {
	pool := &AgentPool{}
	calls := 0
	pool.SetRemoteAgentHTTPClient(delegationHTTPDoer(func(r *http.Request) (*http.Response, error) {
		calls++
		if r.URL.String() != "https://private.invalid/v1/responses" || r.Header.Get("Authorization") != "Bearer secret" {
			t.Fatalf("wrong request: %+v", r)
		}
		if r.Context().Err() != nil {
			return nil, r.Context().Err()
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"output":[{"content":[{"type":"output_text","text":"supplied transport"}]}]}`))}, nil
	}))
	_, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true, RemoteAgents: []RemoteAgent{{Name: "remote", URL: "https://private.invalid", APIKey: "secret"}}})(types.NewJob())
	fragment, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote"})
	if err != nil || len(fragment.Messages) != 1 || fragment.Messages[0].Content != "supplied transport" {
		t.Fatalf("supplied transport result=%+v err=%v", fragment, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := dispatch(ctx, cogito.AgentRunSpec{Type: "remote"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error=%v", err)
	}
	if calls != 2 {
		t.Fatalf("supplied client calls=%d", calls)
	}
	pool.SetRemoteAgentHTTPClient(delegationHTTPDoer(func(*http.Request) (*http.Response, error) { return nil, errors.New("secret private.invalid") }))
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote"}); err == nil || strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), "private.invalid") {
		t.Fatalf("transport error=%v", err)
	}
}

func TestSubAgentResolverRejectsSelfAliasAndNewAmbiguity(t *testing.T) {
	pool := &AgentPool{pool: AgentPoolData{"parent": {}, "a": {}, "b": {}}}
	pool.SetSubAgentResolver(func(_ string, _ *types.Job, candidate string) (string, bool) {
		if candidate == "a" || candidate == "parent" {
			return "self", true
		}
		return "worker", true
	})
	defs, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true})(types.NewJob())
	if len(defs) != 1 || defs[0].Name != "worker" {
		t.Fatalf("self alias exposed: %+v", defs)
	}
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "self"}); err == nil || errors.Is(err, cogito.ErrDispatchFallback) {
		t.Fatalf("self alias dispatched: %v", err)
	}
	pool.SetSubAgentResolver(func(_ string, _ *types.Job, candidate string) (string, bool) {
		if candidate == "parent" {
			return "self", true
		}
		return "worker", true
	})
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "worker"}); err == nil || !strings.Contains(err.Error(), "authorized") {
		t.Fatalf("newly ambiguous alias dispatched: %v", err)
	}
}

func TestRemoteSubAgentPreservesClientRedirectPolicy(t *testing.T) {
	followed := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/responses" {
			http.Redirect(w, r, "/other", http.StatusTemporaryRedirect)
			return
		}
		followed = true
		io.WriteString(w, `{"output":[]}`)
	}))
	defer server.Close()
	pool := &AgentPool{}
	pool.SetRemoteAgentHTTPClient(&http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }})
	_, dispatch := pool.subAgentProvider("parent", &AgentConfig{EnableSubAgents: true, RemoteAgents: []RemoteAgent{{Name: "remote", URL: server.URL}}})(types.NewJob())
	if _, err := dispatch(context.Background(), cogito.AgentRunSpec{Type: "remote"}); err == nil || !strings.Contains(err.Error(), "307") {
		t.Fatalf("redirect policy ignored: %v", err)
	}
	if followed {
		t.Fatal("followed a redirect forbidden by host client")
	}
}
