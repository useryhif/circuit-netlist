# Batch review summary

40 drawings processed, 40 succeeded, 40 passed format validation.

Trust levels come from the components' and nets' own features (no reference netlist is used) and decide what needs manual review:

| level | meaning | drawings |
|---|---|---|
| high | no known risk signal | 001 003 004 008 009 010 011 012 013 015 016 025 026 027 029 030 031 033 034 035 036 040 |
| caution | depends on per-drawing conventions | 014 017 019 020 021 022 024 028 032 037 038 |
| review | contains a known unreliable stage | 002 005 006 007 018 023 039 |

## Drawings that need manual review (review / caution)

- **002**（review）
  - small two-terminal devices (R/C easily confused): R1, R2, R3, R4, R5, R6, R7, R8, R9
  - small BJTs (polarity rule unreliable, box height <45px): Q1, Q2
  - gate/base attached to a supply net (possible marker over-merge): M22.Gate->VSS, Q1.Base->VSS
- **005**（review）
  - small two-terminal devices (R/C easily confused): R1, C1, C2, C3
- **006**（review）
  - small two-terminal devices (R/C easily confused): C1, C2
- **007**（review）
  - 1 component(s) missing a full port mapping: R5
- **018**（review）
  - small BJTs (polarity rule unreliable, box height <45px): Q1, Q2, Q3, Q4
  - gate/base attached to a supply net (possible marker over-merge): M11.Gate->VSS, Q1.Base->VSS, Q2.Base->VSS
- **023**（review）
  - 2 component(s) missing a full port mapping: M1, Q1
  - small two-terminal devices (R/C easily confused): R2, R3, R4
  - small BJTs (polarity rule unreliable, box height <45px): Q1, Q2
  - gate/base attached to a supply net (possible marker over-merge): Q2.Base->VSS
- **039**（review）
  - small two-terminal devices (R/C easily confused): C1
  - small BJTs (polarity rule unreliable, box height <45px): Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20
- **014**（caution）
  - gate/base attached to a supply net (possible marker over-merge): Q1.Base->VSS, Q2.Base->VSS
- **017**（caution）
  - bulk-style devices (reference records Body for some drawings only): M1, M2, M3, M4, M5, M6, M7, M8, M9, M10
  - gate/base attached to a supply net (possible marker over-merge): M3.Gate->VDD
- **019**（caution）
  - gate/base attached to a supply net (possible marker over-merge): Q1.Base->VSS, Q2.Base->VSS
- **020**（caution）
  - bulk-style devices (reference records Body for some drawings only): M1, M2
- **021**（caution）
  - gate/base attached to a supply net (possible marker over-merge): M4.Gate->VSS
- **022**（caution）
  - bulk-style devices (reference records Body for some drawings only): M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11
- **024**（caution）
  - bulk-style devices (reference records Body for some drawings only): M1, M2
- **028**（caution）
  - gate/base attached to a supply net (possible marker over-merge): Q1.Base->VSS, Q2.Base->VSS
- **032**（caution）
  - gate/base attached to a supply net (possible marker over-merge): Q1.Base->VSS, Q2.Base->VSS
- **037**（caution）
  - gate/base attached to a supply net (possible marker over-merge): Q1.Base->VSS, Q2.Base->VSS
- **038**（caution）
  - gate/base attached to a supply net (possible marker over-merge): M13.Gate->VSS, M14.Gate->VSS
