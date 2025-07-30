while true; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        mem_info=$(cat /proc/meminfo | grep -E 'MemFree:')
        echo "$timestamp $mem_info" >> memory_usage.log
        sleep 60
    done