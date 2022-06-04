#!/bin/bash
# USE:
# ./data-extractor.sh <juliet dir> <python_script> <python_interp> <dataset folder>
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
JULIET_DIR="$1"
TESCASESUPPORT_SRC="$JULIET_DIR/testcasesupport/io.c $JULIET_DIR/testcasesupport/std_thread.c"
TESCASESUPPORT_INC="$JULIET_DIR/testcasesupport/"
COMPILER_DEFINES="-DINCLUDEMAIN"
OMIT_BAD="-DOMITBAD"
OMIT_GOOD="-DOMITGOOD"
COMPILER_FLAGS="-pthread -Wno-everything"
PYTHON3="$3"
PYTHON_SCRIPT="$2"

DATASET_DIR="$4"
mkdir -p $DATASET_DIR
rm -rf $DATASET_DIR/*

# as ccsm only accepts c files, we only work with them
EXTENSIONS="*.c"

strip_win_files() {
    echo "${*%%*CWE*@(wchar_t|w32)*.c}"
}

CWE_DIRS="$(ls -d -- $(realpath $1/testcases/CWE*/))"
# CWE_DIRS="/home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcases/CWE415_Double_Free /home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcases/CWE416_Use_After_Free"

# CWE_DIRS="/home/kostr2010/cod/ml-in-stat-analysis/third-party/juliet-test-suite-c/testcases/CWE190_Integer_Overflow"
file_routine() {
    [[ $1 =~ (CWE[0-9]+_[A-Z,a-z,_]+__.+_[0-9]{2}) ]]
    EXECUTABLE_NAME=${BASH_REMATCH[0]}
    CUR_DATASET_DIR=$DATASET_DIR/$EXECUTABLE_NAME/

    if [[ -d "$CUR_DATASET_DIR" ]] ; then
        # echo "$EXECUTABLE_NAME already done"
        return
    fi

    mkdir -p $CUR_DATASET_DIR

    SOURCES="$(find $(realpath "$2") -type f -name "*$EXECUTABLE_NAME*")"

    LOG_FILE_METRICS="$CUR_DATASET_DIR/metrics.csv"
    CMD_METRICS="$TOOL $TOOL_ARGS $DUMP_METRICS $SOURCES -- $TESCASESUPPORT_SRC -I $TESCASESUPPORT_INC $COMPILER_DEFINES $COMPILER_FLAGS"

    LOG_FILE_TOKENS="$CUR_DATASET_DIR/tokens.txt"
    CMD_TOKENS="$TOOL $TOOL_ARGS $DUMP_TOKENS $SOURCES -- $TESCASESUPPORT_SRC -I $TESCASESUPPORT_INC $COMPILER_DEFINES $COMPILER_FLAGS"

    LOG_FILE_ANALYZER_GCC_GOOD="$CUR_DATASET_DIR/analyzer-gcc-good.txt"
    CMD_ANALYZER_GCC_GOOD="$ANALYZER_GCC $SOURCES $TESCASESUPPORT_SRC -I $TESCASESUPPORT_INC $COMPILER_DEFINES $OMIT_BAD $COMPILER_FLAGS"

    LOG_FILE_ANALYZER_GCC_BAD="$CUR_DATASET_DIR/analyzer-gcc-bad.txt"
    CMD_ANALYZER_GCC_BAD="$ANALYZER_GCC $SOURCES $TESCASESUPPORT_SRC -I $TESCASESUPPORT_INC $COMPILER_DEFINES $OMIT_GOOD $COMPILER_FLAGS"

    LOG_FILE_ANALYZER_CPPCHECK_GOOD="$CUR_DATASET_DIR/analyzer-cppcheck-good.xml"
    CMD_ANALYZER_CPPCHECK_GOOD="$ANALYZER_CPPCHECK --output-file="$LOG_FILE_ANALYZER_CPPCHECK_GOOD" -I $TESCASESUPPORT_INC $COMPILER_DEFINES $OMIT_BAD $TESCASESUPPORT_SRC $SOURCES"

    LOG_FILE_ANALYZER_CPPCHECK_BAD="$CUR_DATASET_DIR/analyzer-cppcheck-bad.xml"
    CMD_ANALYZER_CPPCHECK_BAD="$ANALYZER_CPPCHECK --output-file="$LOG_FILE_ANALYZER_CPPCHECK_BAD" -I $TESCASESUPPORT_INC $COMPILER_DEFINES $OMIT_GOOD $TESCASESUPPORT_SRC $SOURCES"

    $CMD_METRICS > $LOG_FILE_METRICS

    $CMD_TOKENS > $LOG_FILE_TOKENS

    $CMD_ANALYZER_GCC_BAD &> $LOG_FILE_ANALYZER_GCC_BAD
    $CMD_ANALYZER_GCC_GOOD &> $LOG_FILE_ANALYZER_GCC_GOOD

    $CMD_ANALYZER_CPPCHECK_BAD &> /dev/null
    $CMD_ANALYZER_CPPCHECK_GOOD &> /dev/null

    $PYTHON3 $PYTHON_SCRIPT $CUR_DATASET_DIR $JULIET_DIR $TOKEN_WINDOW
}

cleanup_routine() {
    [[ $1 =~ (CWE[0-9]+) ]]
    folder=${BASH_REMATCH[0]}

    mkdir -p ${DATASET_DIR}/${folder}
    datasets=$(find $(realpath "$DATASET_DIR/${folder}_*") -type f -name "*data.csv")

    for F in  $datasets; do
        cat "$F" >> "${DATASET_DIR}/${folder}/data.csv"
    done

    mv ${DATASET_DIR}/${folder}_* ${DATASET_DIR}/${folder}/

    echo "FINISHED $folder"
}

N_PROC=120
PROC_COUNTER=0

for DIR in $CWE_DIRS; do
    PROGRESS_COUNTER=0
    echo "STARTED $DIR"

    C_FILES="$(find $(realpath "$DIR") -type f -name "$EXTENSIONS")"
    C_FILES=$(strip_win_files $C_FILES)

    TOTAL_FILES="$(find $(realpath "$DIR") -type f -name "$EXTENSIONS" | wc -l )"
    
    if [[ -z "${C_FILES// }" ]] ; then
        echo "$DIR can't be processed (win only or .cpp only)"
        continue
    fi

    echo "finished preprocessing files for $DIR"

    for F in $C_FILES ; do
        ((PROC_COUNTER = PROC_COUNTER % N_PROC)); ((PROC_COUNTER++ == 0)) && wait
        file_routine "$F" "$DIR" &
        ((PROGRESS_COUNTER++))
        echo "[PROGRESS] $((PROGRESS_COUNTER * 100 / TOTAL_FILES))% ($PROGRESS_COUNTER/$TOTAL_FILES)"
    done

    wait

    cleanup_routine $DIR &


    # for F in $C_FILES ; do
    #     file_routine "$F" "$DIR"
    # done

    # cleanup_routine $DIR
done

wait

echo "JOINING DATASETS"

DATASETS="$(find $(realpath "$DATASET_DIR") -type f -name "*data.csv")"

for F in $DATASETS; do
    cat "$F" >> "$DATASET_DIR/dataset.csv"
done
