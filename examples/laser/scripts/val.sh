#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

# path="${LASER}/checkpoints/laser_lstm/checkpoint1.pt"
path="${LASER}/checkpoints/laser_lstm/checkpoint_last.pt"

generate() {
  cfg_file=$1
  oup_file=$2

  fairseq-generate \
      "${cfg_file}" \
      --user-dir laser_src \
      --task laser \
      --gen-subset valid \
      --path "${path}" \
      --beam 5 --batch-size 128 --remove-bpe | tee "${oup_file}"

  grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
  grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"
}

generate "cfgs/en.json" "output/en.out"

fairseq-score --sys "output/en.out.sys" --ref "output/en.out.ref"
