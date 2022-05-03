from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        rfclf = joblib.load("rfclf.pkl")
        
        # Get values through input bars
        Months_As_Customer = request.form.get("Months As Customer")
        Age = request.form.get("Age")
        Policy_Deductable = request.form.get("Policy Deductable")
        Policy_Annual_Premium = request.form.get("Policy Annual Premium")
        Umbrella_Limit = request.form.get("Umbrella Limit")
        Capital_Gains = request.form.get("Capital Gains")
        Capital_Loss = request.form.get("Capital Loss")
        Incident_Hour = request.form.get("Incident Hour")
        Bodily_Injured = request.form.get("Bodily Injured")
        Witness = request.form.get("Witness")
        Total_Claim_Amount = request.form.get("Total Claim Amount")
        Policy_CSL = request.form.get("Policy CSL")
        Sex = request.form.get("Sex")
        Property_Damage = request.form.get("Property Damage")
        Police_Reported = request.form.get("Police Reported")
        Incident_Type = request.form.get("Incident Type")
        Collision_Type = request.form.get("Collision Type")
        Incident_State = request.form.get("Incident State")
        Authorities_Contacted = request.form.get("Authorities Contacted")
        Policy_State = request.form.get("Policy State")
        Incident_Severity = request.form.get("Incident Severity")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[Months_As_Customer, Age,Policy_Deductable,Policy_Annual_Premium,Umbrella_Limit,Capital_Gains,Capital_Loss,Incident_Hour,
        Bodily_Injured,Witness,Total_Claim_Amount,Policy_CSL,Sex,Property_Damage,Police_Reported,Incident_Type,Collision_Type,Incident_State,Authorities_Contacted,Policy_State,Incident_Severity]], 
        columns = ["Months As Customer", "Age","Policy Deductable","Policy Annual Premium","Umbrella Limit","Capital Gains","Capital Loss","Incident Hour",
        "Bodily Injured","Witness","Total Claim Amount","Policy CSL","Sex","Property Damage","Police Reported","Incident Type","Collision Type","Incident State","Policy State","Incident Severity"])
        
        # Get prediction
        prediction = rfclf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("index.html", results = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)