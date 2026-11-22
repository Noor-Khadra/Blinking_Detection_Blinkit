#!/bin/zsh
exec "${0:A:h}/Run_Dataset_Recorder.command" --session-name rana_v1 --connect --port /dev/cu.BITalino-8C-2A --channel 4 --camera 0
