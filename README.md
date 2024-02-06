# Play with llama
Create pvc for disk space by (already created)
```
kubectl create -f llama2-7b-pvc.yaml
```

Craete a pod with OS, cuda, pytorch etc. 
```
kubectl create -f llama2-7b-pod.yaml
```

Then login to the pod:
```
kubectl exec -it llama2-7b-pod -- /bin/bash
```
Prepara for some tools
```
apt-get update
apt-get install git
apt-get install wget
git clone https://github.com/facebookresearch/llama.git
cd llama
./download.sh
```
Enter the email from Facebook and model you like after you run `./downlead.sh`


# AICode
Dataset: https://github.com/microsoft/CodeXGLUE/tree/main

# Recent work in Code generation
1. https://huggingface.co/blog/starcoder
