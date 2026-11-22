"""Participant prompts for the ten-minute BlinkIT protocol."""
from pathlib import Path
from threading import Thread
import sys

BREAK = 12
COUNTDOWN = 3
START = BREAK + COUNTDOWN
ACTIVE = 60
BLOCK_SECONDS = START + ACTIVE
VIDEO = Path(__file__).with_name("harbor.mp4")
ACTIVITIES = {
    "rest": ("Quiet fixation", "Look at the dot. Sit comfortably and blink naturally."),
    "reading": ("Silent reading", "Read the story silently at your own pace. Blink naturally."),
    "video": ("Watch the boats", "Watch the short harbor animation. Blink naturally."),
    "voluntary": ("Blink on cue", "Each prompt asks for one, two or three blinks. Do it once, then relax."),
    "eye_movements": ("Eye movements", "Keep your head still. Move only your eyes in the direction shown."),
    "head_motion": ("Head movements", "Follow one prompt at a time. Turn gently and blink naturally."),
    "talking": ("Read aloud", "Read the story aloud at your normal volume. Continue until the break."),
    "artifacts": ("Facial movements", "Follow each prompt once. Relax your face between actions."),
    "sustained": ("Hold eyes closed", "Close your eyes when prompted. A high beep means open them again."),
    "mixed": ("Conversation", "Read the question. Follow the prompts to turn, answer aloud and look back."),
}
PASSAGE = (
    "On Saturday morning, Maya walked down to the harbor before the town had fully woken up. "
    "A small blue boat was tied to the last post on the pier. Beside it, a fisherman was painting its name in white letters. "
    "He stopped to watch a paper boat drift past his feet. A little girl ran along the pier, trying to keep up with it. "
    "When the paper boat caught on a rope, Maya reached down and set it free. The girl waved and hurried after it. "
    "By the time Maya reached the market, the stalls were open and the smell of warm bread had filled the street. "
    "She bought two rolls and walked back to see what name the fisherman had chosen."
)
CUES = {
    "voluntary": [(t, msg) for t, msg in zip((3,10,17,24,31,38),
        ("Blink once", "Double blink: blink twice", "Blink once", "Triple blink: blink three times", "Double blink: blink twice", "Blink once"))],
    "head_motion": [(t,msg) for t,msg in zip(range(0,44,4),
        ("Turn your head left","Face forward","Turn your head right","Face forward","Nod down","Face forward","Turn your head right","Face forward","Turn your head left","Face forward","Nod once"))],
    "eye_movements": [(i*4,"Look "+v) for i,v in enumerate(("left","straight ahead","right","straight ahead","up","straight ahead","down","straight ahead","left","straight ahead","right"))],
    "artifacts": [(t,msg) for t,msg in zip((3,10,17,24,31,38),("Raise your eyebrows","Smile","Gently clench your jaw","Raise your eyebrows","Smile","Gently clench your jaw"))],
    "sustained": [(t+d,msg) for t in (5,18,31) for d,msg in ((0,"Close both eyes"),(3,"Open your eyes"))],
    "mixed": [(0,"Read: What did you do this morning?"),(6,"Turn toward your teammate"),(9,"Answer the question aloud"),(18,"Face the screen"),(22,"Read: What would you like to do this weekend?"),(28,"Turn toward your teammate"),(31,"Answer the question aloud"),(42,"Face the screen")],
}
IDLE = {"voluntary":"Relax and blink naturally","artifacts":"Relax your face","sustained":"Keep your eyes comfortably open and blink naturally",
        "rest":"●","reading":"Read silently","talking":"Read aloud","video":"Watch the boats"}


def cue_at(key, second):
    title, instruction = ACTIVITIES[key]
    if second < BREAK:
        return "briefing", 0, instruction
    if second < START:
        tick = int(second)
        return "countdown", tick, str(START - tick)
    scheduled, msg = START, IDLE.get(key, title)
    for offset, action in CUES.get(key, ()):
        at = START + (round((offset - 3) * 4/3) + 3 if key == "sustained" and action == "Open your eyes" else round(offset * 4/3))
        if at > second:
            break
        scheduled, msg = at, action
        if key in ("artifacts","voluntary") and second >= at + 3:
            scheduled, msg = at + 3, IDLE[key]
    return "activity", scheduled, msg


def beep(widget, high=False):
    if sys.platform == "win32":
        import winsound
        Thread(target=winsound.Beep, args=(1250 if high else 730, 280 if high else 145), daemon=True).start()
    else:
        widget.bell()


class ParticipantView:
    def __init__(self, root, stop):
        import tkinter as tk
        import cv2
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.title("BlinkIT · Participant")
        self.window.protocol("WM_DELETE_WINDOW", stop)
        self.video = cv2.VideoCapture(str(VIDEO))
        if not self.video.isOpened():
            self.window.destroy()
            raise ValueError("Could not open the bundled harbor.mp4 video.")
        self.frame_no = -1
        self.window.attributes("-fullscreen", True)
        self.window.bind("<Escape>", lambda event: self.window.attributes("-fullscreen", False))
        self.header = tk.Label(self.window, font=("Arial",18,"bold"), fg="white", pady=14)
        self.header.pack(fill="x")
        self.action = tk.Label(self.window, font=("Arial",32,"bold"), justify="center", fg="#102f38", bg="#f5f8f7")
        self.action.pack(fill="both",expand=True)
        self.media = tk.Label(self.window, font=("Arial",18), justify="left", bg="#f5f8f7", padx=20)
        self.footer = tk.Label(self.window, bg="#f5f8f7", fg="#254850", font=("Arial",15), pady=10)
        self.footer.pack(side="bottom",fill="x")
        self.window.configure(bg="#f5f8f7")
        self.window.bind("<Configure>", self.resize)
        self.window.deiconify()

    def resize(self,event):
        if event.widget == self.window:
            for label in (self.action,self.media):
                label.configure(wraplength=max(320,event.width-90))

    def render(self, key, block, local, phase, prompt):
        import cv2
        from PIL import Image, ImageTk
        title = ACTIVITIES[key][0]
        stage, color = {"briefing":("BREAK · Read the next instruction","#234c73"),
                        "countdown":("Get ready","#87500f"),"activity":("Do this now","#086358")}[phase]
        self.header.configure(text=f"Block {block}/10   ·   {title}   ·   {stage}",bg=color)
        self.action.configure(text=prompt)
        remaining = BLOCK_SECONDS - int(local) if phase == "activity" else START - int(local)
        self.footer.configure(text=f"{remaining} seconds of activity remaining" if phase == "activity" else f"Recording paused · Activity begins in {remaining} seconds")
        if phase=="activity" and key in ("reading","talking","video"):
            self.media.pack(before=self.footer,fill="x",pady=10)
            if key=="video":
                n=int((local-START)*10)
                if n!=self.frame_no:
                    if n!=self.frame_no+1:
                        self.video.set(cv2.CAP_PROP_POS_FRAMES,n)
                    ok,frame=self.video.read()
                    if ok:
                        self.media.image=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                        self.media.configure(image=self.media.image,text="")
                    self.frame_no=n
            else:
                self.media.configure(image="",text=PASSAGE)
        else:
            self.media.pack_forget()

    def close(self):
        self.video.release()
        self.window.destroy()
