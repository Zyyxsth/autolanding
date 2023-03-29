# rostopic pub -1 /sim_cmd/rc/in sim_node/sim_rccmd "sim_rccmd_ch5: 1 
# sim_rccmd_ch6: 1
# sim_rccmd_ch7: 0
# sim_rccmd_ch9: 2" 

rostopic pub -1 /set_task_state sim_msgs/task_state "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
state: 'TRACK'
is_hover_mode: false
state_id: 0" 

