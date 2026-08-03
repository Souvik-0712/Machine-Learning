import customtkinter as ctk
from tkinter import messagebox
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from collections import defaultdict

# ==============================
# AI STUDY PLANNER ENGINE
# ==============================


class StudyPlannerAI:

    def __init__(self):

        self.subjects = []
        self.ml = StudyMLModel()
        self.recommendation = AIRecommendation()
        self.visualizer=StudyVisualizer()

    def calculate_days(self, exam_date):
        today = datetime.today().date()
        return (exam_date - today).days
    
    def add_subject(self, subject):

        self.subjects.append(subject)

    def prioritize_topics(self):

        priority = {"Hard": 1, "Medium": 2, "Easy": 3}

        for subject in self.subjects:

            subject["topics"].sort(key=lambda x: priority[x["difficulty"]])


    def generate_plan(self, exam_date, daily_hours):

        self.prioritize_topics()

        plan = []

        current_date = datetime.today()

        for subject in self.subjects:

            for topic in subject["topics"]:

                remaining_hours = topic["hours"]

                while remaining_hours > 0:

                    if current_date.date() >= exam_date:
                        break

                    study_time = min(daily_hours, remaining_hours)

                    plan.append(
                        {
                            "date": current_date.strftime("%d-%m-%Y"),
                            "subject": subject["name"],
                            "topic": topic["name"],
                            "hours": study_time,
                        }
                    )

                    remaining_hours -= study_time

                    current_date += timedelta(days=1)

        total_topics = 0

        for s in self.subjects:

            total_topics += len(s["topics"])

        prediction = self.ml.predict_daily_hours(
            total_topics, 0, self.calculate_days(exam_date)
        )

        performance = self.ml.predict_performance(
            total_topics, 0, self.calculate_days(exam_date)
        )

        advice = self.recommendation.generate_message(self.subjects, daily_hours)

        return plan, prediction, performance, advice


# ==============================
# MACHINE LEARNING AI MODULE
# ==============================


class StudyMLModel:

    def __init__(self):

        self.hour_model = LinearRegression()

        self.performance_model = DecisionTreeClassifier()

        self.train_models()

    # --------------------------
    # Training sample data
    # --------------------------

    def train_models(self):

        # Input:
        # [total_topics, completed_topics, available_days]

        X = np.array(
            [
                [10, 8, 20],
                [15, 10, 15],
                [20, 18, 30],
                [25, 10, 20],
                [5, 5, 10],
                [30, 20, 25],
            ]
        )

        # Required daily hours

        y_hours = np.array([2, 4, 3, 7, 1, 8])

        self.hour_model.fit(X, y_hours)

        # Performance labels

        # 0 = Low
        # 1 = Medium
        # 2 = High

        y_performance = np.array([2, 1, 2, 0, 2, 0])

        self.performance_model.fit(X, y_performance)

    # --------------------------
    # Predict required hours
    # --------------------------

    def predict_daily_hours(self, total_topics, completed_topics, days):

        result = self.hour_model.predict([[total_topics, completed_topics, days]])

        return round(float(result[0]), 2)

    # --------------------------
    # Predict performance
    # --------------------------

    def predict_performance(self, total_topics, completed_topics, days):

        result = self.performance_model.predict(
            [[total_topics, completed_topics, days]]
        )

        labels = {0: "Low", 1: "Medium", 2: "High"}

        return labels[result[0]]


# ==============================
# AI RECOMMENDATION ENGINE
# ==============================


class AIRecommendation:

    def generate_message(self, subjects, daily_hours):

        total_topics = 0

        hard_topics = 0

        for subject in subjects:

            for topic in subject["topics"]:

                total_topics += 1

                if topic["difficulty"] == "Hard":

                    hard_topics += 1

        message = """



===== AI STUDY ADVICE =====


"""

        if hard_topics > 0:

            message += f"• Focus on {hard_topics} hard topics first.\n"

        if daily_hours < 3:

            message += "• Increase daily study time for better results.\n"

        else:

            message += "• Your daily study schedule is balanced.\n"

        if total_topics > 10:

            message += "• Divide large topics into smaller sessions.\n"

        message += "• Revise completed topics every 3 days.\n"

        message += "• Keep the final days only for revision and mock tests.\n"

        return message



# ==============================
# DATA VISUALIZATION MODULE
# ==============================

class StudyVisualizer:
    def subject_hours_chart(self,plan):
        subject_hours=defaultdict(int)

        for item in plan:
            subject_hours[item["subject"]]+=item["hours"]

        subjects=list(
            subject_hours.keys()
        )

        hours=list(
            subject_hours.values()
        )

        plt.figure(
            figsize=(8,5)
        )
        plt.bar(
            subjects,hours
        )
        plt.xlabel("Subjects")
        plt.ylabel("Study Hours")
        plt.title("Subject Wise Study Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def daily_schedule_chart(self,plan):
        dates=[]
        hours=[]

        for item in plan:
            dates.append(item["date"])
            hours.append(item["hours"])
        plt.figure(figsize=(10,5))
        plt.plot(dates,hours,marker="o")
        plt.xlabel("Date")
        plt.ylabel("Study Hours")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def difficulty_chart(self,subjects):
        difficulty_count={
            "Hard":0,
            "Medium":0,
            "Easy":0
        }

        for subject in subjects:
            for topic in subject["topics"]:
                difficulty_count[
                    topic["difficulty"]
                ]+=1
        labels=list(
            difficulty_count.keys()
        )
        values=list(
            difficulty_count.values()
        )
        plt.figure(figsize=(6,6))
        plt.pie(values,labels=labels,autopct="%1.1f%%")
        plt.title( "Topic Difficulty Distribution")
        plt.show()



# ==============================
# MODERN GUI APPLICATION
# ==============================


class StudyPlannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "AI Study Planner"
        )
        self.root.geometry(
            "1200x700"
        )
        ctk.set_appearance_mode(
            "dark"
        )
        ctk.set_default_color_theme(
            "blue"
        )
        self.ai = StudyPlannerAI()
        self.current_plan=[]
        self.create_gui()

    # ==========================
    # MAIN GUI
    # ==========================
    def create_gui(self):
        # Main container
        self.sidebar = ctk.CTkFrame(
            self.root,
            width=250
        )
        self.sidebar.pack(
            side="left",
            fill="y"
        )
        self.main = ctk.CTkFrame(
            self.root
        )
        self.main.pack(
            side="right",
            expand=True,
            fill="both"
        )

        # Title
        self.title = ctk.CTkLabel(
            self.sidebar,
            text="AI\nStudy Planner",
            font=(
                "Arial",
                28,
                "bold"
            )
        )
        self.title.pack(
            pady=30
        )
        # Sidebar buttons
        ctk.CTkButton(
            self.sidebar,
            text="Generate Plan",
            command=self.generate_plan
        ).pack(
            pady=15,
            padx=20
        )
        ctk.CTkButton(
            self.sidebar,
            text="Subject Chart",
            command=self.show_subject_chart
        ).pack(
            pady=15,
            padx=20
        )
        ctk.CTkButton(
            self.sidebar,
            text="Schedule Chart",
            command=self.show_daily_chart
        ).pack(
            pady=15,
            padx=20
        )
        ctk.CTkButton(
            self.sidebar,
            text="Difficulty Analysis",
            command=self.show_difficulty_chart
        ).pack(
            pady=15,
            padx=20
        )

        # ======================
        # Dashboard
        # ======================


        heading=ctk.CTkLabel(
            self.main,
            text="Create Your AI Study Plan",
            font=(
                "Arial",
                24,
                "bold"
            )
        )
        heading.pack(
            pady=20
        )
        # Input frame
        input_frame=ctk.CTkFrame(
            self.main
        )
        input_frame.pack(
            pady=10
        )
        self.exam_entry=self.create_input(
            input_frame,
            "Exam Date DD-MM-YYYY",
            0
        )
        self.hours_entry=self.create_input(
            input_frame,
            "Daily Study Hours",
            1
        )
        self.subject_entry=self.create_input(
            input_frame,
            "Subject Name",
            2
        )
        self.topic_entry=self.create_input(
            input_frame,
            "Topic Name",
            3
        )
        self.difficulty_box=ctk.CTkComboBox(
            input_frame,
            values=[
                "Hard",
                "Medium",
                "Easy"
            ]
        )
        self.difficulty_box.grid(
            row=4,
            column=1,
            pady=5
        )
        self.difficulty_box.set(
            "Hard"
        )
        self.topic_hours=self.create_input(
            input_frame,
            "Topic Hours",
            5
        )

        ctk.CTkButton(
            input_frame,
            text="Add Topic",
            command=self.add_topic
        ).grid(
            row=6,
            column=1,
            pady=15
        )
        # Output area
        self.output=ctk.CTkTextbox(
            self.main,
            width=850,
            height=280,
            font=(
                "Consolas",
                14
            )
        )
        self.output.pack(
            pady=20
        )

    def create_input(
            self,
            frame,
            placeholder,
            row):
        entry=ctk.CTkEntry(
            frame,
            width=300,
            placeholder_text=placeholder
        )
        entry.grid(
            row=row,
            column=1,
            pady=5
        )
        return entry

    # ======================
    # ADD TOPIC
    # ======================

    def add_topic(self):
        subject=self.subject_entry.get()
        topic=self.topic_entry.get()
        difficulty=self.difficulty_box.get()
        hours=self.topic_hours.get()

        if not subject or not topic:
            messagebox.showerror(
                "Error",
                "Fill all fields"
            )
            return
        found=False
        for s in self.ai.subjects:
            if s["name"]==subject:
                s["topics"].append(
                    {
                    "name":topic,
                    "difficulty":difficulty,
                    "hours":int(hours)
                    }
                )
                found=True

        if not found:
            self.ai.subjects.append(
                {
                "name":subject,
                "topics":[
                    {
                    "name":topic,
                    "difficulty":difficulty,
                    "hours":int(hours)
                    }
                ]
                }
            )
        messagebox.showinfo(
            "Added",
            "Topic Added Successfully"
        )
        self.topic_entry.delete(
            0,
            "end"
        )

    # ======================
    # GENERATE PLAN
    # ======================

    def generate_plan(self):
        exam=datetime.strptime(
            self.exam_entry.get(),
            "%d-%m-%Y"
        ).date()
        hours=int(
            self.hours_entry.get()
        )
        plan,prediction,performance,advice=\
            self.ai.generate_plan(
                exam,
                hours
            )
        self.current_plan=plan
        self.output.delete(
            "0.0",
            "end"
        )
        self.output.insert(
            "end",
            "====== AI GENERATED TIMETABLE ======\n\n"
        )

        for item in plan:
            self.output.insert(
                "end",

                f"""

Date:
{item['date']}

Subject:
{item['subject']}

Topic:
{item['topic']}

Hours:
{item['hours']}


------------------------------

"""

            )



        self.output.insert(

            "end",

            f"""

AI Prediction:

Required Hours:
{prediction}

Performance:
{performance}


{advice}

"""

        )

    def show_subject_chart(self):
        self.ai.visualizer.subject_hours_chart(
            self.current_plan
        )

    def show_daily_chart(self):
        self.ai.visualizer.daily_schedule_chart(
            self.current_plan
        )

    def show_difficulty_chart(self):
        self.ai.visualizer.difficulty_chart(
            self.ai.subjects
        )

# ==============================
# RUN APPLICATION
# ==============================


if __name__ == "__main__":

    root = ctk.CTk()

    app = StudyPlannerApp(root)

    root.mainloop()
