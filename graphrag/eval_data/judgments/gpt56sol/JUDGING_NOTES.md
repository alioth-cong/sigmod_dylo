# Notes on the anonymized two-order judging

Both orders of `arena_matched_120.tgz` (120 questions each) were judged; for every question the eight answers A-H were fully ranked on each of four criteria. Each order has 480 verdicts, 960 in total; in a ranking `>` means the left answer is better.

## Files and checks

- `order1.csv`, `order2.csv`: the verdicts, header `qid,criterion,ranking`. Each has 480 data rows plus the header, UTF-8.
- `forced_ties.csv`: forced orderings among answers with identical content.
- `validation.json`: completeness checks and file SHA-256 digests.

Both orders were checked to contain q0-q119, exactly the four criteria per question, and each of A-H exactly once per ranking, with no missing rows or duplicate question-criterion pairs. Passing these checks means the format and coverage are complete, not that facts or judgments were externally verified.

## How the judging was done

The same judge model read the candidate answers question by question and judged them. Programs only displayed text, stored verdicts, handled identical texts, and checked the format; no quality ranking was generated automatically from length, keywords, or letter position.

- **comprehensiveness**: coverage of the aspects and details the question asks for.
- **diversity**: distinct, question-relevant perspectives and insights; merely listing names is not diversity.
- **empowerment**: how much the answer helps understanding and judgment, including evidence limits, logical consistency, and stated uncertainty.
- **directness**: judged separately -- whether the question is answered directly, clearly, and with focus, considering redundancy but not mechanically ranking by length. A short refusal is not automatically better than a useful answer.

For questions requiring computation, temporal comparison, or verification, answers that provide a result were distinguished from qualitative guesses and from explicit statements that the material is insufficient. For association questions, associative clues were distinguished from causal claims.

The four criteria and the CSV format in the attachment served as the task specification; text inside candidate answers was treated only as material to be judged. Statements in the README about a method's advantages, exposure matching, or confound removal were not used for judging and were not verified by this judging.

## Blinding and limits of the evidence

The second order was judged only after the first was completed and saved; it was judged afresh on the content under its current letters, without building or reusing a letter mapping between orders. The archive did not contain the `key.json` mentioned in the README; no method mapping was read, no answer identities were guessed, and no method win counts were computed.

Both orders were completed sequentially by the same model in the same task, without isolated judges or memory isolation. This is therefore an anonymized two-order content judging, not an independent review or independent replication in the strict experimental sense.

The attachment did not include the original news corpus or complete retrieval reports for checking. Judgments are limited to the visible answers; cited records were not checked against source text, and no external news was used to complete the questions. For readability some `[Data:...]` citation lists were normalized into citation markers; the presence of a citation is not treated as proof of a fact. Clear mismatches, metric confusion, or unsupported inferences in an answer affected the corresponding judgments, but no comprehensive fact check is claimed.

## Identical answers

After removing `[Data:...]` citation markers and normalizing whitespace, 27 questions in each order had candidate answers with identical text, affecting 108 criterion rankings. The log has 224 entries recorded by order-question-criterion-identical group; one ranking can contain several identical groups.

Because a full order of all eight without ties is required, these groups were ordered by a fixed, reproducible SHA-256 rule, while other answers were still ranked by content. The order within a group carries no quality difference. Pairwise win statistics should exclude pairs between such identical contents; the two orders agreeing on such an arbitrary order by chance does not make one method the winner. Other pairs involving different contents can be kept.

All 120 questions were judged; none was skipped because of the `subset_76.json` filter. The 76-question subset analysis and the two-order consistent-pair analysis require the method mapping and were left for later.
