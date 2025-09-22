### Results
- Sentence 10150 (13 tokens):
- WORDS : Those coming from other denominations will welcome the opportunity to become informed.
- GOLD  : DET VERB ADP ADJ NOUN VERB VERB DET NOUN PRT VERB VERB .
- PRED  : DET NOUN ADP ADJ NOUN VERB VERB DET NOUN PRT VERB VERB .
- Acc   : 0.923 | Viterbi path prob: 2.274e-43
---
- Sentence 10151 (18 tokens):
- WORDS : The preparatory class is an introductory face-to-face group in which new members become acquainted with one another .
- GOLD  : DET ADJ NOUN VERB DET ADJ ADJ NOUN ADP DET ADJ NOUN VERB VERB ADP NUM DET .
- PRED  : DET ADJ NOUN VERB DET ADJ NOUN NOUN ADP DET ADJ NOUN VERB VERB ADP NUM NOUN .
- Acc   : 0.889 | Viterbi path prob: 5.374e-60
---
- Sentence 10152 (16 tokens):
- WORDS : It provides a natural transition into the life of the local church and its organizations .
- GOLD  : PRON VERB DET ADJ NOUN ADP DET NOUN ADP DET ADJ NOUN CONJ DET NOUN .
- PRED  : PRON VERB DET ADJ NOUN ADP DET NOUN ADP DET ADJ NOUN CONJ DET NOUN .
- Acc   : 1.000 | Viterbi path prob: 1.847e-42
---
- Overall micro accuracy on [10150,10153): 0.936

### Why the tagger may fail on certain tokens

- **Ambiguity & limited context**: First‑order HMMs only condition on the previous tag (bigram), so longer‑range syntactic constraints are ignored.
- **Emission sparsity**: Even with add‑1, rare words (especially names, numbers, punctuation variants) rely heavily on UNK emissions, which are not tag‑specific enough.
- **Lowercasing trade‑off**: Helps with sparsity but removes capitalization cues for PROPN vs NOUN, often hurting proper nouns and sentence‑initial tokens.
- **Domain specifics**: Brown’s genres vary; first 10k sentences may bias π, A, B toward their tag distributions, which may differ from the eval slice’s genre.
- **Universal tagset mapping**: Collapses fine distinctions; some near‑ties in the trellis can flip with small smoothing differences.

### Why the tagger succeeds on others

- **Strong local patterns**: Determiner → Noun (DET→NOUN), Preposition → Determiner (ADP→DET), Pronoun → Verb (PRON→VERB) are frequent and well‑captured by A.
- **High‑frequency function words**: Emissions like the/DET, of/ADP, to/PRT (or PART) receive strong B probabilities, steering the path during decoding.