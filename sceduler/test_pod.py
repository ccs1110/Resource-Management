apiVersion: apps/v1
kind: Deployment
metadata:
  name: yfh1
  labels:
    app: yfh1
spec:
  replicas: 1
  selector: # define how the deployment finds the pods it mangages
    matchLabels:
      app: yfh1
  template: # define the pods specifications
    metadata:
      labels:
        app: yfh1
        appointedge: "true"
        needgpu: "true"
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nodetype
                operator: In
                values:
                - edge
      containers:
      - command: ["sh", "-c", "tail -f /dev/null"]
        name: yfh1
        image: yfh1:v0.1
        resources:
          limits:
            cpu: 200m
            memory: 200Mi
          requests:
            cpu: 200m
            memory: 200Mi
      schedulerName: myscheduler
