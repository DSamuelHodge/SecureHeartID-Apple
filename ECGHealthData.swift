import Foundation
import HealthKit

class ECGHealthData {
    
    // MARK: - Properties
    
    static let shared = ECGHealthData()
    static let healthStore: HKHealthStore = HKHealthStore()
    
    // MARK: - Data Types
    
    static var readDataTypes: [HKSampleType] {
        return allECGDataTypes
    }
    
    static var shareDataTypes: [HKSampleType] {
        return allECGDataTypes
    }
    
    private static var allECGDataTypes: [HKSampleType] {
        let typeIdentifiers: [String] = [
            HKObjectType.electrocardiogramType().identifier
        ]
        
        return typeIdentifiers.compactMap { getSampleType(for: $0) }
    }
    
    // MARK: - Authorization
    
    func requestHealthDataAccessIfNeeded(completion: @escaping (_ success: Bool) -> Void) {
        let readDataTypes = Set(ECGHealthData.allECGDataTypes)
        let shareDataTypes = Set(ECGHealthData.allECGDataTypes)
        
        requestHealthDataAccessIfNeeded(toShare: shareDataTypes, read: readDataTypes, completion: completion)
    }
    
    private func requestHealthDataAccessIfNeeded(toShare shareTypes: Set<HKSampleType>?,
                                                 read readTypes: Set<HKObjectType>?,
                                                 completion: @escaping (_ success: Bool) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            print("Health data is not available!")
            completion(false)
            return
        }
        
        print("Requesting HealthKit authorization for ECG data...")
        ECGHealthData.healthStore.requestAuthorization(toShare: shareTypes, read: readTypes) { (success, error) in
            if let error = error {
                print("requestAuthorization error:", error.localizedDescription)
            }
            
            if success {
                print("HealthKit ECG authorization request was successful!")
            } else {
                print("HealthKit ECG authorization was not successful.")
            }
            
            completion(success)
        }
    }
    
    // MARK: - ECG Data Fetching
    
    func fetchECGData(startDate: Date = Date().addingTimeInterval(-30 * 24 * 60 * 60), // Last 30 days
                      endDate: Date = Date(),
                      completion: @escaping ([HKElectrocardiogram]?, Error?) -> Void) {
        let ecgType = HKObjectType.electrocardiogramType()
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        
        let query = HKSampleQuery(sampleType: ecgType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sortDescriptor]) { (query, samples, error) in
            guard let ecgSamples = samples as? [HKElectrocardiogram] else {
                completion(nil, error)
                return
            }
            
            completion(ecgSamples, nil)
        }
        
        ECGHealthData.healthStore.execute(query)
    }
    
    // MARK: - ECG Voltage Data Fetching
    
    func fetchECGVoltageData(for ecgSample: HKElectrocardiogram, completion: @escaping ([HKQuantitySample]?, Error?) -> Void) {
        var voltageMeasurements: [HKQuantitySample] = []
        
        let query = HKElectrocardiogramQuery(ecgSample) { (query, result) in
            switch result {
            case .measurement(let measurement):
                if let voltageQuantity = measurement.quantity(for: .appleWatchSimilarToLeadI) {
                    let voltageSample = HKQuantitySample(type: HKQuantityType.quantityType(forIdentifier: .electrocardiogramVoltageMeasurement)!,
                                                         quantity: voltageQuantity,
                                                         start: measurement.timeSample,
                                                         end: measurement.timeSample)
                    voltageMeasurements.append(voltageSample)
                }
            case .done:
                completion(voltageMeasurements, nil)
            case .error(let error):
                completion(nil, error)
            @unknown default:
                completion(nil, NSError(domain: "ECGHealthData", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unknown case in ECG voltage query result"]))
            }
        }
        
        ECGHealthData.healthStore.execute(query)
    }
    
    // MARK: - Helper Functions
    
    private static func getSampleType(for identifier: String) -> HKSampleType? {
        if let quantityType = HKObjectType.quantityType(forIdentifier: HKQuantityTypeIdentifier(rawValue: identifier)) {
            return quantityType
        } else if let categoryType = HKObjectType.categoryType(forIdentifier: HKCategoryTypeIdentifier(rawValue: identifier)) {
            return categoryType
        } else if identifier == HKObjectType.electrocardiogramType().identifier {
            return HKObjectType.electrocardiogramType()
        }
        return nil
    }
}

// MARK: - ECG Classification Extension

extension HKElectrocardiogram.Classification {
    var description: String {
        switch self {
        case .notSet:
            return "Not Set"
        case .sinusRhythm:
            return "Sinus Rhythm"
        case .atrialFibrillation:
            return "Atrial Fibrillation"
        case .inconclusiveLowHeartRate:
            return "Inconclusive (Low Heart Rate)"
        case .inconclusiveHighHeartRate:
            return "Inconclusive (High Heart Rate)"
        case .inconclusivePoorReading:
            return "Inconclusive (Poor Reading)"
        case .inconclusiveOther:
            return "Inconclusive (Other)"
        case .unrecognized:
            return "Unrecognized"
        @unknown default:
            return "Unknown"
        }
    }
}