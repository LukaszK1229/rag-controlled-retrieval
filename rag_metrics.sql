SELECT COUNT(*) AS queries
FROM rag_logs
WHERE event = 'query_received';

SELECT
  ROUND(
    COUNT(*) FILTER (WHERE event = 'query_stopped') * 100.0 /
    NULLIF(COUNT(*) FILTER (WHERE event = 'query_received'), 0),
    2
  ) AS refusal_pct
FROM rag_logs;


SELECT
  json_extract(payload, '$.reason') AS reason,
  COUNT(*) AS cnt
FROM rag_logs
WHERE event = 'query_stopped'
GROUP BY reason
ORDER BY cnt DESC;


SELECT
  ROUND(
    COUNT(*) FILTER (
      WHERE event = 'retrieval_results'
        AND json_array_length(json_extract(payload, '$.top_k')) > 0
    ) * 100.0 /
    NULLIF(COUNT(*) FILTER (WHERE event = 'query_received'), 0),
    2
  ) AS retrieval_coverage_pct
FROM rag_logs;


WITH scores AS (
  SELECT
    json_extract(j.value, '$.score') AS score
  FROM rag_logs,
       json_each(json_extract(payload, '$.top_k')) AS j
  WHERE event = 'retrieval_results'
)
SELECT
  ROUND(AVG(score), 4) AS avg_score,
  ROUND(MIN(score), 4) AS best_score,
  ROUND(MAX(score), 4) AS worst_score
FROM scores;


SELECT
  COUNT(*) FILTER (WHERE event = 'answer_generated') AS answers,
  COUNT(*) FILTER (WHERE event = 'query_received') AS queries
FROM rag_logs;


SELECT
  ROUND(AVG(
    CAST(json_extract(payload, '$.answer_length') AS INTEGER)
  ), 1) AS avg_answer_length
FROM rag_logs
WHERE event = 'answer_generated';


SELECT
  json_extract(j.value, '$.chunk_id') AS chunk_id,
  COUNT(*) AS hits
FROM rag_logs,
     json_each(json_extract(payload, '$.top_k')) AS j
WHERE event = 'retrieval_results'
GROUP BY chunk_id
ORDER BY hits DESC
LIMIT 10;


SELECT COUNT(*) AS blocked_after_retrieval
FROM rag_logs
WHERE event = 'query_stopped'
  AND json_extract(payload, '$.reason') = 'low_confidence';
