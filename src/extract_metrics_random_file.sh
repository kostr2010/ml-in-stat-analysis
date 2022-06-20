#!/bin/bash
# USE:
# ./data-extractor.sh file.c
shopt -s extglob

##################
# HYPER PARAMETERS
TOKEN_WINDOW=25
##################

TOOL="ccsm"
ANALYZER_GCC="gcc-10 -fanalyzer -Wanalyzer-too-complex"
ANALYZER_CPPCHECK="cppcheck --std=c99 --max-ctu-depth=10 --xml"
TOOL_ARGS="--exclude-std-headers --disable-file --disable-global"
DUMP_TOKENS="--dump-tokens --output-format=none"
DUMP_METRICS="--output-format=csv"
PYTHON3="python3"
CURRENT_DIR="$(dirname "$0")"
PYTHON_SCRIPT="${CURRENT_DIR}/extract_metrics_random_file.py"
SOURCES="$@"
HASH="$((0x$(sha1sum <<< "$SOURCES")0))"

DATASET_DIR="$PWD/data-$HASH/"
mkdir -p $DATASET_DIR
rm -rf $DATASET_DIR/*

LOG_FILE_METRICS="$DATASET_DIR/metrics.csv"
CMD_METRICS="$TOOL $TOOL_ARGS $DUMP_METRICS $SOURCES -- -DUNCLUDEMAIN -I /home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport"

LOG_FILE_TOKENS="$DATASET_DIR/tokens.txt"
CMD_TOKENS="$TOOL $TOOL_ARGS $DUMP_TOKENS $SOURCES -- -DUNCLUDEMAIN -I /home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport/"

LOG_FILE_ANALYZER_GCC="$DATASET_DIR/analyzer-gcc.txt"
CMD_ANALYZER_GCC="$ANALYZER_GCC -DUNCLUDEMAIN $SOURCES  -I /home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport/"

LOG_FILE_ANALYZER_CPPCHECK="$DATASET_DIR/analyzer-cppcheck.xml"
CMD_ANALYZER_CPPCHECK="$ANALYZER_CPPCHECK --output-file="$LOG_FILE_ANALYZER_CPPCHECK" -DUNCLUDEMAIN -I /home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcasesupport/ $SOURCES"

$CMD_METRICS > $LOG_FILE_METRICS

$CMD_TOKENS > $LOG_FILE_TOKENS

$CMD_ANALYZER_GCC &> $LOG_FILE_ANALYZER_GCC

$CMD_ANALYZER_CPPCHECK &> /dev/null

$PYTHON3 $PYTHON_SCRIPT $DATASET_DIR $TOKEN_WINDOW