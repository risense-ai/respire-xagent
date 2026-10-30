//go:build integration

// Optional live-contract bridge: copy into tests/integration of the audited
// risense-ai/respire-memsrv c39bc874 checkout. It reuses that repo's isolated
// fixture-only harness; never point its database settings at real data.
package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestXagentAdapterBridgeFixture(t *testing.T) {
	h := setup(t)
	seedPatchTree(t, h)
	ctx := context.Background()
	if _, err := h.db.Pool.Exec(ctx, `update users set external_id='31' where id=1;
 update users set external_id='33' where id=3;
 update api_keys set scopes=array['memory:read','memory:write'] where name='fixture-service';
 update notes set metadata=metadata || '{"user_id":"31","execution_scope_client_application_id":"42","xagent_contract_fixture":true}'::jsonb`); err != nil {
		t.Fatal(err)
	}
	root := os.Getenv("XAGENT_CONTRACT_BRIDGE_DIR")
	if root == "" {
		t.Fatal("XAGENT_CONTRACT_BRIDGE_DIR required")
	}
	initial := patchStructure(t, h)
	if err := os.WriteFile(filepath.Join(root, "url"), []byte(h.url), 0600); err != nil {
		t.Fatal(err)
	}
	wait := func(name string) {
		t.Helper()
		deadline := time.Now().Add(3 * time.Minute)
		for time.Now().Before(deadline) {
			if _, err := os.Stat(filepath.Join(root, name)); err == nil {
				return
			}
			time.Sleep(20 * time.Millisecond)
		}
		t.Fatalf("bridge did not receive %s", name)
	}
	wait("ordinary-done")
	if patchStructure(t, h) != initial {
		t.Fatal("ordinary adapter edits changed notes/edges structure")
	}
	// A separate authorized writer reparents after the Python client read. This
	// control request intentionally uses legacy structural PUT; adapter traffic
	// must contain PATCH only and must not restore the old parent/path.
	status, body := h.request("PUT", "/v1/memory/notes/patch-child", "fixture-service", "31", `{"parent_id":"patch-extra","content":"control reparent","metadata":{"user_id":"31","execution_scope_client_application_id":"42","execution_scope_agent":10}}`)
	expect(t, status, 200, body)
	moved := patchStructure(t, h)
	if err := os.WriteFile(filepath.Join(root, "moved"), []byte("ready"), 0600); err != nil {
		t.Fatal(err)
	}
	wait("done")
	if patchStructure(t, h) != moved {
		t.Fatal("stale adapter edit restored or changed moved structure/edges")
	}
}
