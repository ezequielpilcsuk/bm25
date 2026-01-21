package bm25

import (
	"fmt"
	"strings"
	"sync"
	"testing"
)

// TestIDFRaceCondition demonstrates the race condition in the IDF function
// when called concurrently from multiple goroutines.
// Run with: go test -race -run TestIDFRaceCondition
func TestIDFRaceCondition(t *testing.T) {
	// Create a corpus with many documents to increase the chance of race conditions
	corpus := make([]string, 1000)
	for i := 0; i < 1000; i++ {
		corpus[i] = fmt.Sprintf("document %d with various terms like alpha beta gamma delta epsilon", i)
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	bm25Index, err := NewBM25Okapi(corpus, tokenizer, 1.2, 0.75, nil)
	if err != nil {
		t.Fatalf("Failed to create BM25 index: %v", err)
	}

	// Simulate concurrent queries - this will trigger the race condition
	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Use unique terms to force idfCache writes (not just reads)
			query := []string{"unique", "term", fmt.Sprintf("query%d", i)}
			docIDs := []int{0, 1, 2, 3, 4}

			// This call triggers concurrent access to idfCache in the IDF function
			_, err := bm25Index.GetBatchScoresParallel(query, docIDs, bm25Index)
			if err != nil {
				t.Errorf("GetBatchScoresParallel error: %v", err)
			}
		}(i)
	}
	wg.Wait()
}

// TestGetScoresParallelRaceCondition tests for race conditions in GetScoresParallel
// Run with: go test -race -run TestGetScoresParallelRaceCondition
func TestGetScoresParallelRaceCondition(t *testing.T) {
	corpus := make([]string, 500)
	for i := 0; i < 500; i++ {
		corpus[i] = fmt.Sprintf("document %d containing words like foo bar baz qux", i)
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	bm25Index, err := NewBM25Okapi(corpus, tokenizer, 1.2, 0.75, nil)
	if err != nil {
		t.Fatalf("Failed to create BM25 index: %v", err)
	}

	var wg sync.WaitGroup
	numGoroutines := 50

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Each goroutine uses different query terms to maximize cache writes
			query := []string{
				fmt.Sprintf("term%d", i),
				fmt.Sprintf("word%d", i),
				fmt.Sprintf("search%d", i),
			}

			_, err := bm25Index.GetScoresParallel(query, bm25Index)
			if err != nil {
				t.Errorf("GetScoresParallel error: %v", err)
			}
		}(i)
	}
	wg.Wait()
}

// TestConcurrentIDFAccess directly tests concurrent IDF calls
// Run with: go test -race -run TestConcurrentIDFAccess
func TestConcurrentIDFAccess(t *testing.T) {
	corpus := make([]string, 100)
	for i := 0; i < 100; i++ {
		corpus[i] = fmt.Sprintf("doc %d alpha beta gamma", i)
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	bm25Index, err := NewBM25Okapi(corpus, tokenizer, 1.2, 0.75, nil)
	if err != nil {
		t.Fatalf("Failed to create BM25 index: %v", err)
	}

	var wg sync.WaitGroup
	numGoroutines := 200

	// Directly call IDF from multiple goroutines with uncached terms
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Mix of cached (alpha, beta) and uncached terms
			terms := []string{
				"alpha",
				"beta",
				fmt.Sprintf("uncached%d", i),
			}
			for _, term := range terms {
				_, _ = bm25Index.IDF(term)
			}
		}(i)
	}
	wg.Wait()
}
