# RC-ZOQO ICASSP 2027 Draft V0.6

This version is a writing-style refactor of V0.5. No technical claim, theorem result, experiment result, limitation, counterexample, or numerical result from V0.5 has been removed.

Main changes:
- added a Background and Problem Setup section before the theory;
- introduced explicit Definition / Assumption / Lemma structure;
- separated the two long theorem arguments into named supporting lemmas;
- moved detailed proofs to an Appendix, following the presentation style of theory-oriented ICASSP papers;
- retained the exact floor laws, grid-floor bounds, bit law, two failure examples, range-resolution proposition, audit guarantee, false-stop/query bounds, resolution-invariance result, one-grid optimality proposition, all theory-validation numbers, MNIST INT4/INT8 results, and scope/limitations;
- kept four pages of technical content plus a fifth references-only page.

Build:
```bash
pdflatex main.tex
pdflatex main.tex
```
