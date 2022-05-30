# GENERAL NOTES

## ANALYZERS:

- GCC (GCC is a compiler, but like all compilers it statically analyzes the code, plus it outputs warnings (when run with parameters such as these: ?[-fanalyzer]? −Wall −Wextra −Wpointer −arith −Wstrict −prototypes −Wformat −security) that can be mapped (as regular expressions) to code flaw taxonomies, CERT. Cert c coding standard (wiki). https://wiki.sei.cmu.edu/confluence/display/c/GCC.)

- gcc-10 -fanalyzer <file>
