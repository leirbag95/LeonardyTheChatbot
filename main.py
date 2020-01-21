#Creating GUI with tkinter
import tkinter
from tkinter import *
import chatbot
from textblob import TextBlob
from chatbot import Leonardy, Supervised

# init of chatbot leonardy
import sys
isNeedTraining=False
if "--train" in sys.argv:
	isNeedTraining = True
leoBot = chatbot.Leonardy(isNeedTraining)



def send():
	msg = EntryBox.get("1.0",'end-1c').strip()
	EntryBox.delete("0.0",END)
	if msg != '':
		msg_tmp = str(TextBlob(msg).correct())
		ChatLog.config(state=NORMAL)
		ChatLog.insert(END, "You: " + msg_tmp + '\n\n')
		ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
		res, _ = leoBot.get_response(msg_tmp)
		ChatLog.insert(END, "Bot: " + str(res) + '\n\n')
		ChatLog.config(state=DISABLED)
		ChatLog.yview(END)

def send_with_score():
	msg = EntryBox.get("1.0",'end-1c').strip()
	EntryBox.delete("0.0",END)
	if msg != '':
		msg_tmp = str(TextBlob(msg).correct())
		ChatLog.config(state=NORMAL)
		ChatLog.insert(END, "You: " + msg_tmp + '\n\n')
		ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
		res, score = leoBot.get_response(msg_tmp)

		ChatLog.insert(END, "Bot: " + str(res) + " \n (probability: "+str(score)+" )"+ '\n\n')
		ChatLog.config(state=DISABLED)
		ChatLog.yview(END)
base = Tk()
base.title("Leonardy the Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
					bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
					command= send,relief="groove")
SendButtonWithScore = Button(base, font=("Verdana",12,'bold'), text="SWS", width="12", height=5,
					bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
					command= send_with_score,relief="groove")
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial", borderwidth=1, relief="groove")
# EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=430, height=10)
SendButtonWithScore.place(x=6, y=450, height=10)
base.mainloop()
