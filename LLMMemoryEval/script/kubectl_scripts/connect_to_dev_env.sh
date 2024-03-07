#!/bin/bash

type=$1

kubectl exec -it mem-eval-pod-$type -- /bin/bash
