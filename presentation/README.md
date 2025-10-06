# Beamer Presentation: Optimizing LLM Training with Layer-Wise FP8

## Overview
Clean, professional presentation reporting thesis progress on layer-wise FP8 format assignment for efficient LLM training. Features improved layout with integrated figures and results.

## Structure (9 Slides)

1. **Title Slide** - Title, author, institution, university logo
2. **Problem & Motivation** - Memory/compute challenges + FP8 architecture diagram
3. **Proposed Approach** - Layer-wise specialization strategy + visual diagram
4. **Experimental Setup** - Models, dataset, baselines, hardware specs
5. **Results: Performance** - Training time + memory improvements with chart
6. **Results: Loss Variance** - Numerical stability analysis with variance chart
7. **Results: Convergence** - Loss convergence quality with loss curve
8. **Contributions & Impact** - Scientific, practical, and environmental impact
9. **Conclusion & Future Work** - Summary, limitations, future directions

## Key Features

✨ **Clean Layout**: Redesigned with `[squeeze]` option for optimal spacing
📊 **Integrated Figures**: Training time, stability, loss convergence charts
🎨 **Professional Theme**: Madrid theme with beaver color scheme
📝 **Modular Structure**: Each slide in separate file for easy editing

## Building the Presentation

### Quick Build
```bash
cd presentation
pdflatex main.tex
pdflatex main.tex  # Run twice for proper cross-references
```

### With latexmk (recommended)
```bash
cd presentation
latexmk -pdf main.tex
```

### Output
- **File**: `main.pdf`
- **Size**: ~2.3 MB
- **Pages**: 9 slides (16:9 aspect ratio)

## Included Figures

All figures are in `figures/` directory:
- `fp8_convert.png` - Layer-wise FP8 architecture
- `training_time.png` - Training time comparison chart
- `numeric_stability.png` - Loss variance comparison
- `avg_loss.png` - Average loss convergence
- `uet.jpg` - University logo

## Slide Files Structure

```
presentation/
├── main.tex                    # Main document
├── figures/                    # All images
│   ├── fp8_convert.png
│   ├── training_time.png
│   ├── numeric_stability.png
│   ├── avg_loss.png
│   └── uet.jpg
└── slides/                     # Individual slides
    ├── 01_title.tex
    ├── 02_problem.tex
    ├── 03_approach.tex
    ├── 04_setup.tex
    ├── 05_results_performance.tex
    ├── 06_results_stability.tex
    ├── 07_results_convergence.tex
    ├── 08_contributions.tex
    └── 09_conclusion.tex
```

## Key Results Highlighted

- **3B Model**: 10% memory reduction (82.4→74.2 GB), 42% faster training
- **8B Model**: 27% faster training (9.1h→6.6h)
- **Stability**: 50% lower loss variance vs Hybrid FP8
- **Quality**: Maintained perplexity (1.30-1.32 range)

## Customization Tips

### Adjust Slide Spacing
Add `[squeeze]` or `[shrink]` to frame options:
```latex
\begin{frame}[squeeze]{Title}
```

### Change Colors
Edit `main.tex` custom colors:
```latex
\definecolor{darkblue}{RGB}{0,51,102}
\definecolor{lightblue}{RGB}{51,153,255}
```

### Modify Theme
```latex
\usetheme{Madrid}        % Try: Berlin, Copenhagen, Warsaw
\usecolortheme{beaver}   % Try: crane, dolphin, orchid
```

### Add Animations
Insert `\pause` for step-by-step reveals:
```latex
\begin{itemize}
    \item First point \pause
    \item Second point \pause
    \item Third point
\end{itemize}
```

## Presentation Tips

⏱️ **Timing**: ~2-3 minutes per slide = 18-25 minute total
🎯 **Focus**: Emphasize key results on slides 5-7 (performance, stability, convergence)
📊 **Visuals**: One figure per results slide for clarity
❓ **Questions**: Prepare for stability & scalability questions

## Recent Updates

✅ **Separated multi-figure slides**: Split stability results into two focused slides:
   - Slide 6: Loss Variance Analysis (numeric_stability.png)
   - Slide 7: Convergence Quality (avg_loss.png)
   - Better focus and clearer visual presentation
