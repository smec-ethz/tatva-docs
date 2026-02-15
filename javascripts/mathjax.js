window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] }, // Explicitly add the package
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
