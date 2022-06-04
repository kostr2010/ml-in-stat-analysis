#! /bin/bash

CC=clang
ANALYZER=scan-build-9

INCLUDE=/home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport/
SRC1=/home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport/io.c 
SRC2=/home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport/std_thread.c

${ANALYZER} ${CC} -DINCLUDEMAIN -pthread $1 ${SRC1} ${SRC2} -I ${INCLUDE}
