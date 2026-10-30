# 真机记忆测试与修复指南

一句话：跑 `scripts/live_memory_roundtrip.py`，它把 adapter 与 memsrv 的记忆面逐项打一遍；当前预期 **2 项 FAIL**，按本文第三节修完即全 PASS。

## 1 脚本做什么

`scripts/live_memory_roundtrip.py` 用仓库里的 `RemoteMemoryStore` 直接打一个真实 memsrv，不经过 Web 层。逐项检查：

| 检查 | 说明 |
|---|---|
| service health | `GET /healthz` |
| readiness | `GET /readyz`，打印 embedding 的 dim / model / degraded |
| identity | 身份是否已开户（`users.external_id` 能对上 `XAGENT_MEMORY_IDENTITY`） |
| add / get / search / list_all / delete | 正常读写路径 |
| update | **R1**：必须走非结构 PATCH |
| create / delete probe user | 只在 `--create-probe-user` 时执行 |

它只写一条 `live-probe-<时间戳>` 的笔记，结束后删除；不碰其他用户的数据。退出码：全过 0，有 FAIL 为 1，可直接进 CI。

## 2 怎么跑

前置：memsrv 已启动（默认 `http://127.0.0.1:8200`），本地在 xagent 仓库根目录。

```bash
export XAGENT_MEMORY_SERVICE_URL=http://127.0.0.1:8200
export XAGENT_MEMORY_SERVICE_KEY=sk-...        # 服务 key
export XAGENT_MEMORY_IDENTITY=9001             # 平台用户 id，作为 X-User-External-Id

# 方式 A：服务里还没有这个身份，临时建一个探针用户（key 需 admin/org 权限）
python scripts/live_memory_roundtrip.py --create-probe-user

# 方式 B：身份已开户，直接用
python scripts/live_memory_roundtrip.py
```

其它开关：`--json` 输出机读结果；`--keep-note` 保留探针笔记（默认删除）。

当前一次真实输出（2026-10-30，本机 memsrv）：

```
PASS  service health                   HTTP 200
PASS  readiness                        dim=1024 model=bge-m3 degraded=False
PASS  create probe user                user id=881
PASS  add                              live-probe-1793342329
PASS  get                              live probe: memory round-trip (safe to d
PASS  search                           hits=1 ids=['live-probe-1793342329']
PASS  list_all                         notes=1
FAIL  update (R1 nonstructural PATCH)  ... HTTP 405 -- service has no PATCH route; add PATCH /v1/memory/notes/{id} (R1)
PASS  delete
FAIL  delete probe user                HTTP 500 -- audit trail blocks deletion (return 409 or anonymise); row disabled instead (HTTP 200)

8 passed, 2 failed
```

## 3 两个 FAIL 怎么修（memsrv 侧）

### 3.1 缺 PATCH 路由（R1 的最后一环）

适配器 `update` 已改为发 `PATCH /v1/memory/notes/{id}`，载荷只含被编辑的字段（正文、keywords、tags、category、mime_type、timestamp、metadata）。服务端没有该路由，回 405。

要实现的行为：

- 只改请求体里出现的字段；**省略 `parent_id` / `path` / `space_kind` / `space_id` 即不动结构**（这正是 R1 要的：改正文不改变父子关系）。
- 请求体里带了 `parent_id` 就按结构变更处理（与 PUT 一致）。
- 成功返回 `{"success": true, "id": "<note id>", 可选 owner_id / space_kind / space_id / modified_by / degraded}`；适配器会校验 `success` 与 `id`。
- 403 / 404 语义与 PUT 一致（越权与不存在都回 not-found-or-denied 的既有文案）。
- 内容变了重算向量；重算失败保留旧向量（与 PUT 同规则），不要写成 NULL。
- 鉴权、审计、ACL 复用 PUT 的路径。

验收：脚本 `update (R1 nonstructural PATCH)` 一项 PASS。

### 3.2 删用户回 500

`DELETE /v1/org/users/{id}` 对已有审计记录的用户直接 500（`audit_logs_actor_user_id_fkey` 挡住删除）。二选一：

- 返回 **409** 并说明"该用户有审计轨迹，无法物理删除"；或
- 审计脱敏（`actor_user_id` 置空/改成匿名 id），然后允许删除。

不要 500。脚本在 500 时会退化为把该用户 `status=disabled`，并继续报 FAIL。

验收：脚本 `delete probe user` 一项 PASS。

### 3.3 external_id 全空（R4 / R6 的落地缺口）

现网 `users.external_id` 全是 NULL，而适配器用 `X-User-External-Id`（= 平台用户 id）映射身份，映射不上就没有身份可用。

- 开户 / 导入流程必须回填 `users.external_id`，值取平台 xagent 的用户 id。
- `GET /v1/org/users` 已有 `external_id` 字段，核对时直接看它。

验收：不带 `--create-probe-user` 也能过 `identity provisioned`。

## 4 修完怎么算过

1. 脚本全 PASS（`--create-probe-user` 跑一遍，10 项全绿，退出码 0）。
2. 单测不退：

```bash
PYTHONPATH=src pytest tests/core/memory tests/core/agent/test_memory_tool.py \
  tests/web/test_memory_api.py tests/web/test_remote_memory_backend.py \
  tests/web/services/test_memory_policy.py -q
```

（当前基线：978 passed, 1 skipped。跳过的是需要 Go 桥的 live PATCH 契约用例，配好桥后可一起跑。）

## 5 与 R4 / R5 的关系

| 项 | 谁修 | 验收方式 |
|---|---|---|
| R1 非结构更新 | memsrv：补 PATCH | 本脚本 `update` |
| R2 readiness | 已合（PR#2） | 本脚本 `readiness` |
| R3 列表截断 | 已合（PR#1） | 单测 |
| R4 兼容主干 + 按用户选库 | xagent 侧另单（从 `upstream/main` 起的新分支） | 单测 + 本脚本 |
| R5 LanceDB 迁移 | xagent 侧脚本 + memsrv 批量导入端点（另单） | 迁移五项判据 |

本脚本是 R1/R4 的共同验收器：任何一侧改坏，`add→get→search→list→update→delete` 会立刻掉一项。
