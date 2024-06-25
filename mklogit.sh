#!/bin/bash

# create a subdirectory stdev/ and logit/stdev/ under every directory in ./vectors/

for dir in ./vectors/*/
do
    dir=${dir%*/}
    mkdir -p $dir/logit/stdev
    mkdir -p $dir/stdev
done

for dir in ./normalized_vectors/*/
do
    dir=${dir%*/}
    mkdir -p $dir/logit/stdev
    mkdir -p $dir/stdev
done

# for dir in ./erasers/*/
# do
#     dir=${dir%*/}
#     mkdir -p $dir/logit
# done