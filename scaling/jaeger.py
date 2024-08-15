import requests
import time
from datetime import datetime, timedelta
from util.MS_trace import MS_trace
from scaling import *
# Jaeger API endpoint
jaeger_host = 'localhost'
jaeger_port = 30410
service_name = 'compose-post-service'
# Calculate start and end time in microseconds
# Construct the URL and parameters
url = f'http://{jaeger_host}:{jaeger_port}/api/traces'
# Send the request


def get_weight(Microservices_trace):
    print(time.time())
    end_time = int(time.time() * 1e6)
    start_time = int((datetime.now() - timedelta(minutes=1)).timestamp() * 1e6)
    params = {
        'service': service_name,
        'start': start_time,
        'end': end_time
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        traces = response.json().get('data', [])
        # Process and print the traces as a list
        print(f'trace length in one minite: {len(traces)}')
        trace_len=len(traces)
        duration_total = 0
        average_end_latency=0
        for trace in traces:
            trace_duration = 0
            spans = trace['spans']
            process= trace['processes']
            for span in spans:
                trace_duration=max(trace_duration,span['duration'])
                duration_total+=span['duration']
                ms_now=Microservices_trace[process[span['processID']]["serviceName"]]
                ms_now.duration+=span['duration'] 
            average_end_latency+=trace_duration

        for ms in  Microservices_trace.values():
            print(f'{ms.name} duration:{ms.duration}')
            ms.weight=ms.duration/duration_total if duration_total>0 else 0
        average_end_latency=average_end_latency/trace_len 
        return Microservices_trace,average_end_latency
            # ms.clear_duration()
    else:
        print(
            f'Failed to retrieve traces: {response.status_code} - {response.text}')



