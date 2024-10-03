# AppleWatch HeartID - Questions & Queries 
#### Date: October 1, 2024



**Q. Requirement:  What type of information you want to store in iCloud. i.e secure storage.**
1. We plan to store only the encrypted biometric templates in iCloud. No raw ECG data will be stored in the cloud for security reasons.

**Q. Requirement:  Do you wish to pair the Phone with Watch for application? Since the sensors will gather information, but regarding storage and other UI, we expect it to land on iOS devices as**

2. Yes, we want the Apple Watch app to pair with the iPhone. The Watch will collect sensor data, while the iPhone will handle more complex processing, storage, and provide an extended user interface.

**Q. Requirement: What will happen next, once the data is collected from the device ?**

3. Once collected, the ECG data will be immediately processed on-device to generate a secure biometric template. The raw data will then be discarded.

**Q. General Question: Standalone app or part of larger ecosystem?**

4. HeartID will be a standalone app, but with potential for future integration into a larger ecosystem of secure biometric solutions.

**Q. General Question: Minimum supported watchOS version?**

5. The minimum supported watchOS version will be watchOS 7.0 to ensure wide compatibility while maintaining access to necessary APIs.

**Q. General Question:  Which Apple Watch models to support?**

6. We'll support Apple Watch Series 4 and later, as these models have the ECG sensor required for HeartID.

**Q. General Question:  How will user enrollment work?**

7. User enrollment will involve recording multiple ECG samples over a set period (e.g., 5 recordings over 5 days) to account for natural variations.

**Q. General Question:  How often should identity verification occur?**

8. Identity verification should occur at user-initiated security events (e.g., app login, transaction approval) and optionally at regular intervals (configurable by the user or admin).

**Q. General Question: What are the key performance requirements?**

9. Key performance requirements: template generation under 5 seconds, verification under 2 seconds, false accept rate <0.1%, false reject rate <1%.

**Q. General Question:  How to handle errors and provide user feedback?**

10. Errors will be logged locally and, if user-permitted, sent to our servers for analysis. User feedback will be provided via watch haptics and on-screen messages, with more detailed information available on the paired iPhone.

**Q. General Question:  What are the data retention policies and privacy requirements?**

11. Biometric templates will be retained until manually deleted by the user or after 1 year of inactivity. All processing complies with GDPR, CCPA, and HIPAA regulations where applicable.
