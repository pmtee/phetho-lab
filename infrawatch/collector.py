import psutil
import pandas as pd
import datetime
import os

def collect_metrics():
    now = datetime.datetime.now()
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()

    metrics = {
        'timestamp':     now.strftime('%Y-%m-%d %H:%M:%S'),
        'date':          now.strftime('%Y-%m-%d'),
        'hour':          now.hour,
        'cpu_percent':   cpu,
        'ram_percent':   ram.percent,
        'ram_total_gb':  round(ram.total / (1024**3), 2),
        'ram_used_gb':   round((ram.total - ram.available) / (1024**3), 2),
        'disk_percent':  disk.percent,
        'disk_total_gb': round(disk.total / (1024**3), 2),
        'disk_used_gb':  round(disk.used / (1024**3), 2),
        'net_sent_mb':   round(net.bytes_sent / (1024**2), 2),
        'net_recv_mb':   round(net.bytes_recv / (1024**2), 2),
    }
    return metrics

def save_metrics(metrics):
    filepath = 'data/metrics.csv'
    df_new = pd.DataFrame([metrics])
    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filepath, mode='w', header=True, index=False)
    return filepath

def check_alerts(metrics):
    alerts = []
    if metrics['cpu_percent'] > 90:
        alerts.append(f"CRITICAL: CPU at {metrics['cpu_percent']}%")
    elif metrics['cpu_percent'] > 80:
        alerts.append(f"WARNING: CPU at {metrics['cpu_percent']}%")
    if metrics['ram_percent'] > 90:
        alerts.append(f"CRITICAL: RAM at {metrics['ram_percent']}%")
    elif metrics['ram_percent'] > 80:
        alerts.append(f"WARNING: RAM at {metrics['ram_percent']}%")
    if metrics['disk_percent'] > 90:
        alerts.append(f"CRITICAL: Disk at {metrics['disk_percent']}%")
    elif metrics['disk_percent'] > 80:
        alerts.append(f"WARNING: Disk at {metrics['disk_percent']}%")
    return alerts

def run_collection():
    print(f"\n{'='*50}")
    print(f"Collecting metrics...")
    metrics = collect_metrics()
    filepath = save_metrics(metrics)
    print(f"Timestamp:  {metrics['timestamp']}")
    print(f"CPU:        {metrics['cpu_percent']}%")
    print(f"RAM:        {metrics['ram_percent']}%")
    print(f"Disk:       {metrics['disk_percent']}%")
    print(f"Net Sent:   {metrics['net_sent_mb']} MB")
    print(f"Net Recv:   {metrics['net_recv_mb']} MB")
    print(f"Saved to:   {filepath}")
    alerts = check_alerts(metrics)
    if alerts:
        print("\nALERTS:")
        for alert in alerts:
            print(f"  ! {alert}")
    else:
        print("Status:     ALL SYSTEMS HEALTHY")
    print(f"{'='*50}")
    return metrics

if __name__ == '__main__':
    print("InfraWatch Collector starting...")
    run_collection()
