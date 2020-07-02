# Dental Practice Patterns Anomaly Detection
The main goal of this project is to identify dentists that provide sub-optimal care to patients to increase their revenue.

When tooth decay is identified the priority is to preserve as much of the tooth as possible. Applying a filling is the normal first procedure choice. If the tooth decay is too great to apply a filling, but the bottom half of the tooth above the gum line is still viable, the next course of action is to apply a crown. If the tooth is not viable and a crown cannot be applied, then the last course of action is to extract the tooth. This project seeks to identify dentists whose practice patterns appear to be overly prescriptive towards the more extreme solutions. Keep in mind that the more extreme solutions tend to pay more than the simple. Dentists charge more for crowns than fillings.

Key practice patterns that trying to understand:

1. *Crowns to Filling Ratio:* A pattern of a high ratio of crowns to fillings indicates that the dentist may be making little effort to preserve as much as possible of the tooth and is too quick to apply a crown (for which they get paid a lot more). 

2. *Root Canals to Crown Ratio:* A pattern of a high ratio of root canals to crowns indicates that the dentist may be doing unnecessary root canals. It is an indication that the tooth is usually too far decayed for a crown to be the appropriate restoration. An extraction might be the better solution.

3. Extract to Crown Ratio. A pattern of a high ratio of extracts to crown indicates that the dentist may be too quick to resort extracting the tooth (with potential to charge for an implant).

4. High Average Charge Per Visit. A high pattern of charge per visit would tend to indicate that the dentist is focused on performing procedures that generate large fees.  A high amount combined with the first three patterns would help confirm the anomalous patterns may be driven by fee seeking.