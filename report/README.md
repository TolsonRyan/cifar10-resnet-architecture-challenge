# AAAI 报告源文件使用说明

`aaai_report.tex` 是依据真实实验编写的英文课程报告初稿，按 AAAI 2026 官方 LaTeX 模板的包与双栏排版接口编写，配套 `references.bib` 和 `figures/`。

**当前没有生成或声称通过排版检查的 PDF。** 本机未安装 TeX 引擎；AAAI 官方 author-kit 入口返回 HTTP 403，未将未知来源的样式文件冒充官方文件。该源码需要官方 `aaai2026.sty` 和 `aaai2026.bst`。

官方模板出处：

- [AAAI author kit](https://aaai.org/authorkit26/)
- [AAAI 在 Overleaf 的 2026 模板](https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj)（作者为 AAAI；模板标记 CC BY 4.0）

可在 Overleaf 中打开该官方模板，把本目录的 `.tex`、`.bib` 与 `figures/` 放入模板中样式文件所在的 LaTeX 目录，并选择 `aaai_report.tex` 为主文件。此操作需用户登录 Overleaf；当前没有创建或上传任何云端项目。

本地已有官方样式与 TeX 发行版时，可以在此目录执行：

```powershell
pdflatex aaai_report.tex
bibtex aaai_report
pdflatex aaai_report.tex
pdflatex aaai_report.tex
```

作者和队名已按用户确认填写为 Sheng Tao，团队为单人。源码使用 `aaai2026` 的 `draft` 模式，意图在课程初稿中显示作者并不加入会议版权声明；这不是匿名会议投稿。取得官方样式后仍需实际编译检查作者显示、双栏、图表、引用、溢出与页数，并以老师最终要求为准。没有获取新的课程 grading-policy 文件，因此未臆测页数或最新模板年份要求。

`methods_notes.md` 提供中文结果解释、复现证据索引和待填写项，便于团队自行理解和修改。不能将尚未核验的团队注册、GitHub 发布或 Kaggle 成绩写成已完成。
