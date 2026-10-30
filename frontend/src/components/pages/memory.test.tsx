import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/i18n-context", () => {
  const labels: Record<string, string> = {
    "memory.sidebar.contextSource": "Context Source",
    "memory.sidebar.allMemories": "All Memories",
    "memory.sidebar.byCategory": "By Category",
    "memory.sidebar.byTag": "By Tag",
    "memory.sidebar.tagPlaceholder": "Filter by tag",
    "memory.form.categoryLabel": "Category",
    "memory.storeInfo.lastCall": "Last upstream call at {time}",
    "memory.storeInfo.noDetail": "The backend reported no reason",
    "memory.filters.similarity.label": "Similarity Threshold",
    "memory.filters.similarity.strict": "Strict",
    "memory.filters.similarity.lenient": "Lenient",
    "memory.filters.similarity.tooltip": "Higher values match more loosely",
    "memory.scope.org": "Organization scope",
    "memory.scope.visible": "Accessible memories",
    "memory.scope.personal": "My memories",
    "memory.item.owner": "user #{id}",
  }
  const dynamicLabels: Record<string, string> = {
    "memory.storeInfo.states.ready": "Memory store ready",
    "memory.storeInfo.states.degraded": "Memory store degraded",
    "memory.storeInfo.states.unavailable": "Memory store unavailable",
  }
  const interpolate = (value: string, vars?: Record<string, string | number>) =>
    vars
      ? value.replace(/\{(\w+)\}/g, (_match, name: string) => String(vars[name] ?? `{${name}}`))
      : value

  return {
    useI18n: () => ({
      locale: "en",
      setLocale: vi.fn(),
      t: (key: string, vars?: Record<string, string | number>) =>
        interpolate(labels[key] ?? key, vars),
      tDynamic: (key: string, fallback: string, vars?: Record<string, string | number>) =>
        interpolate(dynamicLabels[key] ?? fallback, vars),
    }),
  }
})

import { MemoryPage } from "./memory"

const storeInfoReady = {
  store_type: "RemoteMemoryStore",
  state: "ready",
  detail: "上游最近一次调用成功（HTTP 200）：/v1/memory/stats，15:54:46",
  mode: "vector",
  supports_vector_search: true,
  similarity_threshold: 0.4,
}

const statsPayload = {
  total_count: 4,
  category_counts: { general: 3, test: 1 },
  tag_counts: { w03: 1, e2e: 2 },
  memory_store_type: "remote",
  error: null,
}

const memoriesPayload = {
  memories: [
    {
      id: "m-test",
      content: "probe marker",
      keywords: [],
      tags: ["w03"],
      category: "test",
      timestamp: "2026-10-29T07:55:20Z",
      mime_type: "text/plain",
      metadata: {},
    },
    {
      id: "m-general",
      content: "user marker",
      keywords: ["budget"],
      tags: ["e2e"],
      category: "general",
      timestamp: "2026-10-29T07:50:00Z",
      mime_type: "text/plain",
      metadata: {},
    },
  ],
  total_count: 2,
  filters_used: {},
}

const jsonResponse = (payload: unknown, ok = true) =>
  ({ ok, status: ok ? 200 : 503, json: async () => payload }) as unknown as Response

function mockBackend(options: {
  storeInfo?: unknown | "throw" | "fail"
  memories?: unknown
  stats?: unknown
} = {}) {
  apiRequestMock.mockImplementation(async (url: string) => {
    if (url.includes("/api/memory/store-info")) {
      if (options.storeInfo === "throw") throw new Error("store-info transport failure")
      if (options.storeInfo === "fail") return jsonResponse({ detail: "nope" }, false)
      return jsonResponse(options.storeInfo ?? storeInfoReady)
    }
    if (url.includes("/api/memory/stats")) {
      return jsonResponse(options.stats ?? statsPayload)
    }
    if (url.includes("/api/memory/list")) {
      return jsonResponse(options.memories ?? memoriesPayload)
    }
    throw new Error(`unexpected request: ${url}`)
  })
}

const requestedUrls = () => apiRequestMock.mock.calls.map(call => String(call[0]))
const listUrls = () => requestedUrls().filter(url => url.includes("/api/memory/list"))

function searchThreshold(url: string): number | null {
  const raw = new URL(url).searchParams.get("similarity_threshold")
  return raw === null ? null : Number(raw)
}

async function searchFor(term: string) {
  const searchBox = screen.getByPlaceholderText("memory.filters.search.placeholder")
  fireEvent.change(searchBox, { target: { value: term } })
  await waitFor(
    () => expect(listUrls().some(url => url.includes(`search=${term}`))).toBe(true),
    { timeout: 4000 },
  )
  return listUrls().filter(url => url.includes(`search=${term}`)).at(-1)!
}

beforeEach(() => {
  apiRequestMock.mockReset()
})

afterEach(() => {
  cleanup()
})

describe("MemoryPage", () => {
  it("does not send a similarity threshold on a plain list request", async () => {
    mockBackend()
    render(<MemoryPage />)

    await waitFor(() => expect(listUrls().length).toBeGreaterThan(0))

    expect(listUrls()[0]).not.toContain("similarity_threshold")
    expect(requestedUrls().some(url => url.includes("/api/memory/store-info"))).toBe(true)
  })

  it("sends the store-info threshold only when a search is present", async () => {
    mockBackend()
    render(<MemoryPage />)

    await waitFor(() => expect(listUrls().length).toBeGreaterThan(0))

    const searchBox = screen.getByPlaceholderText("memory.filters.search.placeholder")
    fireEvent.change(searchBox, { target: { value: "zzz" } })

    await waitFor(
      () =>
        expect(
          listUrls().some(
            url => url.includes("search=zzz") && url.includes("similarity_threshold=0.4"),
          ),
        ).toBe(true),
      { timeout: 4000 },
    )
  })

  it("renders sidebar entries from live stats, including categories the page never hard-coded", async () => {
    mockBackend()
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByRole("button", { name: /test/ })).toBeTruthy())

    const allMemories = screen.getByRole("button", { name: /All Memories/ })
    expect(within(allMemories).getByText("4")).toBeTruthy()

    const testCategory = screen.getByRole("button", { name: /test/ })
    expect(within(testCategory).getByText("1")).toBeTruthy()

    // The tag filter is fed by stats.tag_counts.
    expect(screen.getByText("By Tag")).toBeTruthy()
  })

  it("filters the list by the tags picked from stats.tag_counts", async () => {
    mockBackend()
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByText("Filter by tag")).toBeTruthy())
    fireEvent.click(screen.getByText("Filter by tag"))

    const w03Option = await screen.findByRole("button", { name: /^w03/ })
    fireEvent.click(w03Option)

    await waitFor(() => expect(listUrls().some(url => url.includes("tags=w03"))).toBe(true))
  })

  it("shows a ready status strip with the last upstream call time", async () => {
    mockBackend()
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByRole("status")).toBeTruthy())

    const strip = screen.getByRole("status")
    expect(strip.textContent).toContain("Memory store ready")
    expect(strip.textContent).toContain("15:54:46")
  })

  it("renders a degraded status strip with the reported reason", async () => {
    mockBackend({
      storeInfo: {
        ...storeInfoReady,
        state: "degraded",
        detail: "上游最近一次调用失败（HTTP 500），16:01:02",
      },
    })
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByRole("status")).toBeTruthy())

    const strip = screen.getByRole("status")
    expect(strip.textContent).toContain("Memory store degraded")
    expect(strip.textContent).toContain("HTTP 500")
  })

  it("hides the status strip instead of breaking the page when store-info fails", async () => {
    mockBackend({ storeInfo: "throw" })
    render(<MemoryPage />)

    await waitFor(() => expect(listUrls().length).toBeGreaterThan(0))
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalled())

    expect(screen.queryByRole("status")).toBeNull()
    expect(screen.getByText("probe marker")).toBeTruthy()
  })

  it("clamps the legacy store threshold to the measured ceiling", async () => {
    mockBackend({ storeInfo: { ...storeInfoReady, similarity_threshold: 1.5 } })
    render(<MemoryPage />)

    // The store still reports the legacy 1.5; the measured ceiling is 0.40.
    await waitFor(() => expect(screen.getByText("0.40")).toBeTruthy())

    const url = await searchFor("zzz")
    expect(searchThreshold(url)).toBeCloseTo(0.4, 5)
  })

  it("falls back to 0.40 when store-info carries no usable threshold", async () => {
    mockBackend({ storeInfo: {} })
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByText("0.40")).toBeTruthy())

    const url = await searchFor("zzz")
    expect(searchThreshold(url)).toBeCloseTo(0.4, 5)
  })

  it("keeps a store threshold below the ceiling untouched", async () => {
    mockBackend({ storeInfo: { ...storeInfoReady, similarity_threshold: 0.3 } })
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByText("0.30")).toBeTruthy())

    const url = await searchFor("zzz")
    expect(searchThreshold(url)).toBeCloseTo(0.3, 5)
  })

  const ownedMemories = {
    ...memoriesPayload,
    memories: [
      { ...memoriesPayload.memories[0], metadata: { user_id: 2 } },
      memoriesPayload.memories[1],
    ],
  }

  it("badges the org scope and labels each item with its owner", async () => {
    mockBackend({ stats: { ...statsPayload, scope: "org" }, memories: ownedMemories })
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByText("Organization scope")).toBeTruthy())
    expect(screen.getByText("user #2")).toBeTruthy()
  })

  it("labels remote ACL visibility without claiming organization administration", async () => {
    mockBackend({ stats: { ...statsPayload, scope: "visible" }, memories: ownedMemories })
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByText("Accessible memories")).toBeTruthy())
    expect(screen.queryByText("Organization scope")).toBeNull()
  })

  it("shows neither the org badge nor owner labels in the personal scope", async () => {
    mockBackend({ stats: { ...statsPayload, scope: "personal" }, memories: ownedMemories })
    render(<MemoryPage />)

    await waitFor(() => expect(screen.getByText("probe marker")).toBeTruthy())
    expect(screen.queryByText("Organization scope")).toBeNull()
    expect(screen.queryByText("user #2")).toBeNull()
  })
})
