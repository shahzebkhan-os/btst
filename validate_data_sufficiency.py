"""
validate_data_sufficiency.py — Validate if current data meets minimum requirements for ML/NN training.

This script checks all data sources and provides a detailed report on data sufficiency
for training the Temporal Fusion Transformer ensemble model.

Usage:
    python validate_data_sufficiency.py
    python validate_data_sufficiency.py --verbose
    python validate_data_sufficiency.py --export-report
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse

# Try importing pandas, provide helpful error if not available
try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Run: pip install pandas")
    sys.exit(1)

from config import (
    DATA_DIR, BHAVCOPY_DIR, VIX_FILE, FII_FILE,
    TRAIN_WINDOW_DAYS, MIN_HISTORY_ROWS, SYMBOLS
)


class DataSufficiencyValidator:
    """Validates if data meets minimum requirements for ML training."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.warnings = []
        self.errors = []

        # Minimum requirements
        self.MIN_BHAVCOPY_ROWS = TRAIN_WINDOW_DAYS  # 504 days minimum
        self.MIN_VIX_ROWS = TRAIN_WINDOW_DAYS  # 504 days minimum
        self.MIN_FII_DII_ROWS = TRAIN_WINDOW_DAYS  # 504 days minimum
        self.MIN_GLOBAL_ROWS = 50_000  # 500 days × 100 tickers
        self.MIN_OPTION_CHAIN_ROWS = 100_000  # Strike-level historical data

        # Feature requirements
        self.NUM_FEATURES = 75  # 75+ engineered features
        self.MIN_SAMPLES_PER_FEATURE = 10
        self.TFT_PARAMETERS = 150_000  # Estimated trainable parameters
        self.SAMPLES_PER_PARAMETER = 5

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp}] [{level}]"

        if level == "ERROR":
            self.errors.append(message)
            print(f"{prefix} ❌ {message}")
        elif level == "WARNING":
            self.warnings.append(message)
            print(f"{prefix} ⚠️  {message}")
        elif level == "SUCCESS":
            print(f"{prefix} ✅ {message}")
        else:
            if self.verbose:
                print(f"{prefix} {message}")

    def check_bhavcopy_data(self) -> Dict:
        """Check NSE Bhavcopy data sufficiency."""
        self.log("Checking NSE Bhavcopy data...", "INFO")

        result = {
            "name": "NSE Bhavcopy",
            "status": "UNKNOWN",
            "rows": 0,
            "days_coverage": 0,
            "sufficient": False,
            "details": []
        }

        try:
            # Count CSV files in bhavcopy directory
            if not BHAVCOPY_DIR.exists():
                self.log(f"Bhavcopy directory not found: {BHAVCOPY_DIR}", "ERROR")
                result["status"] = "MISSING"
                result["details"].append("Directory does not exist")
                return result

            csv_files = list(BHAVCOPY_DIR.glob("*.csv"))
            result["rows"] = len(csv_files)

            if result["rows"] == 0:
                self.log("No bhavcopy files found", "ERROR")
                result["status"] = "EMPTY"
                result["details"].append("No CSV files found in directory")
                return result

            # Read one file to check structure
            sample_df = pd.read_csv(csv_files[0])
            result["details"].append(f"Sample file columns: {list(sample_df.columns)}")
            result["details"].append(f"Sample file rows: {len(sample_df)}")

            # Estimate total rows across all files
            total_rows = len(csv_files) * len(sample_df)
            result["rows"] = total_rows
            result["days_coverage"] = len(csv_files)

            # Check sufficiency
            if result["days_coverage"] >= self.MIN_BHAVCOPY_ROWS:
                result["status"] = "SUFFICIENT"
                result["sufficient"] = True
                self.log(f"Bhavcopy data SUFFICIENT: {result['days_coverage']} days (>{self.MIN_BHAVCOPY_ROWS} required)", "SUCCESS")
            else:
                result["status"] = "INSUFFICIENT"
                result["sufficient"] = False
                gap = self.MIN_BHAVCOPY_ROWS - result["days_coverage"]
                self.log(f"Bhavcopy data INSUFFICIENT: {result['days_coverage']} days, need {gap} more days", "WARNING")

        except Exception as e:
            self.log(f"Error checking bhavcopy data: {e}", "ERROR")
            result["status"] = "ERROR"
            result["details"].append(f"Error: {str(e)}")

        self.results["bhavcopy"] = result
        return result

    def check_vix_data(self) -> Dict:
        """Check India VIX data sufficiency."""
        self.log("Checking India VIX data...", "INFO")

        result = {
            "name": "India VIX",
            "status": "UNKNOWN",
            "rows": 0,
            "days_coverage": 0,
            "sufficient": False,
            "details": []
        }

        try:
            if not VIX_FILE.exists():
                self.log(f"VIX file not found: {VIX_FILE}", "ERROR")
                result["status"] = "MISSING"
                result["details"].append("File does not exist")
                return result

            # Read VIX file
            df = pd.read_csv(VIX_FILE)
            result["rows"] = len(df)
            result["days_coverage"] = len(df)
            result["details"].append(f"Columns: {list(df.columns)}")
            result["details"].append(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")

            # Check sufficiency
            if result["rows"] >= self.MIN_VIX_ROWS:
                result["status"] = "SUFFICIENT"
                result["sufficient"] = True
                self.log(f"VIX data SUFFICIENT: {result['rows']} rows (>{self.MIN_VIX_ROWS} required)", "SUCCESS")
            else:
                result["status"] = "INSUFFICIENT"
                result["sufficient"] = False
                gap = self.MIN_VIX_ROWS - result["rows"]
                self.log(f"VIX data INSUFFICIENT: {result['rows']} rows, need {gap} more rows", "WARNING")

        except Exception as e:
            self.log(f"Error checking VIX data: {e}", "ERROR")
            result["status"] = "ERROR"
            result["details"].append(f"Error: {str(e)}")

        self.results["vix"] = result
        return result

    def check_fii_dii_data(self) -> Dict:
        """Check FII/DII data sufficiency."""
        self.log("Checking FII/DII data...", "INFO")

        result = {
            "name": "FII/DII Flows",
            "status": "UNKNOWN",
            "rows": 0,
            "days_coverage": 0,
            "sufficient": False,
            "details": []
        }

        try:
            if not FII_FILE.exists():
                self.log(f"FII/DII file not found: {FII_FILE}", "ERROR")
                result["status"] = "MISSING"
                result["details"].append("File does not exist")
                return result

            # Read FII/DII file
            df = pd.read_csv(FII_FILE)
            result["rows"] = len(df)
            result["days_coverage"] = len(df)
            result["details"].append(f"Columns: {list(df.columns)}")

            if result["rows"] > 0:
                result["details"].append(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")

            # Check sufficiency (CRITICAL feature)
            if result["rows"] >= self.MIN_FII_DII_ROWS:
                result["status"] = "SUFFICIENT"
                result["sufficient"] = True
                self.log(f"FII/DII data SUFFICIENT: {result['rows']} rows (>{self.MIN_FII_DII_ROWS} required)", "SUCCESS")
            elif result["rows"] < 10:
                result["status"] = "CRITICAL"
                result["sufficient"] = False
                self.log(f"FII/DII data CRITICALLY INSUFFICIENT: {result['rows']} rows, need {self.MIN_FII_DII_ROWS} rows", "ERROR")
                self.log("FII/DII is a critical feature for model accuracy. Missing data will degrade performance by ~10%", "ERROR")
            else:
                result["status"] = "INSUFFICIENT"
                result["sufficient"] = False
                gap = self.MIN_FII_DII_ROWS - result["rows"]
                self.log(f"FII/DII data INSUFFICIENT: {result['rows']} rows, need {gap} more rows", "WARNING")

        except Exception as e:
            self.log(f"Error checking FII/DII data: {e}", "ERROR")
            result["status"] = "ERROR"
            result["details"].append(f"Error: {str(e)}")

        self.results["fii_dii"] = result
        return result

    def check_global_markets_data(self) -> Dict:
        """Check global markets data sufficiency."""
        self.log("Checking Global Markets data...", "INFO")

        result = {
            "name": "Global Markets",
            "status": "UNKNOWN",
            "rows": 0,
            "tickers": 0,
            "days_per_ticker": 0,
            "sufficient": False,
            "details": []
        }

        try:
            extended_dir = DATA_DIR / "extended"
            parquet_file = extended_dir / "market_data_extended.parquet"

            if not parquet_file.exists():
                self.log(f"Global markets file not found: {parquet_file}", "WARNING")
                result["status"] = "MISSING"
                result["details"].append("File does not exist (will be auto-fetched on first run)")
                result["details"].append("Run: python market_data_extended.py --start-date 2020-01-01")
                return result

            # Read parquet file
            df = pd.read_parquet(parquet_file)
            result["rows"] = len(df)

            # Count unique tickers
            if 'symbol' in df.columns:
                result["tickers"] = df['symbol'].nunique()
                result["days_per_ticker"] = result["rows"] // result["tickers"] if result["tickers"] > 0 else 0
                result["details"].append(f"Unique tickers: {result['tickers']}")
                result["details"].append(f"Average days per ticker: {result['days_per_ticker']}")

            # Check sufficiency
            if result["rows"] >= self.MIN_GLOBAL_ROWS:
                result["status"] = "SUFFICIENT"
                result["sufficient"] = True
                self.log(f"Global markets data SUFFICIENT: {result['rows']} rows (>{self.MIN_GLOBAL_ROWS} required)", "SUCCESS")
            else:
                result["status"] = "INSUFFICIENT"
                result["sufficient"] = False
                gap = self.MIN_GLOBAL_ROWS - result["rows"]
                self.log(f"Global markets data INSUFFICIENT: {result['rows']} rows, need {gap} more rows", "WARNING")
                self.log(f"Target: {self.MIN_GLOBAL_ROWS} rows (500 days × 100 tickers)", "WARNING")

        except Exception as e:
            self.log(f"Error checking global markets data: {e}", "ERROR")
            result["status"] = "ERROR"
            result["details"].append(f"Error: {str(e)}")

        self.results["global_markets"] = result
        return result

    def calculate_ml_requirements(self) -> Dict:
        """Calculate if data meets ML model requirements."""
        self.log("Calculating ML model requirements...", "INFO")

        # Get bhavcopy data
        bhavcopy = self.results.get("bhavcopy", {})
        total_rows = bhavcopy.get("rows", 0)

        # Calculate sequences (rows - lookback window)
        lookback_window = 20  # TFT_MAX_ENCODER_LENGTH
        usable_sequences = max(0, total_rows - lookback_window)

        # Feature-based requirement
        min_sequences_features = self.NUM_FEATURES * self.MIN_SAMPLES_PER_FEATURE
        feature_ratio = usable_sequences / min_sequences_features if min_sequences_features > 0 else 0

        # Parameter-based requirement
        min_sequences_params = self.TFT_PARAMETERS * self.SAMPLES_PER_PARAMETER
        param_ratio = usable_sequences / min_sequences_params if min_sequences_params > 0 else 0

        result = {
            "total_rows": total_rows,
            "usable_sequences": usable_sequences,
            "num_features": self.NUM_FEATURES,
            "tft_parameters": self.TFT_PARAMETERS,
            "min_sequences_features": min_sequences_features,
            "min_sequences_params": min_sequences_params,
            "feature_requirement_ratio": feature_ratio,
            "parameter_requirement_ratio": param_ratio,
            "feature_based_sufficient": feature_ratio >= 1.0,
            "parameter_based_sufficient": param_ratio >= 1.0,
            "overall_assessment": "UNKNOWN"
        }

        # Overall assessment
        if feature_ratio >= 1.0 and param_ratio >= 0.05:  # At least 5% of ideal parameters
            result["overall_assessment"] = "SUFFICIENT"
            self.log(f"ML requirements: SUFFICIENT for training", "SUCCESS")
            self.log(f"  - Feature-based: {usable_sequences:,} / {min_sequences_features:,} = {feature_ratio:.2f}x", "SUCCESS")
            self.log(f"  - Parameter-based: {usable_sequences:,} / {min_sequences_params:,} = {param_ratio:.3f}x (with regularization)", "INFO")
        elif feature_ratio >= 1.0:
            result["overall_assessment"] = "MARGINAL"
            self.log(f"ML requirements: MARGINAL (sufficient for features, regularization needed)", "WARNING")
            self.log(f"  - Feature-based: {usable_sequences:,} / {min_sequences_features:,} = {feature_ratio:.2f}x", "SUCCESS")
            self.log(f"  - Parameter-based: {usable_sequences:,} / {min_sequences_params:,} = {param_ratio:.3f}x (low)", "WARNING")
        else:
            result["overall_assessment"] = "INSUFFICIENT"
            self.log(f"ML requirements: INSUFFICIENT", "ERROR")
            self.log(f"  - Feature-based: {usable_sequences:,} / {min_sequences_features:,} = {feature_ratio:.2f}x", "ERROR")
            self.log(f"  - Parameter-based: {usable_sequences:,} / {min_sequences_params:,} = {param_ratio:.3f}x", "ERROR")

        self.results["ml_requirements"] = result
        return result

    def generate_report(self) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("ML DATA SUFFICIENCY VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary table
        report.append("DATA SOURCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Data Source':<25} {'Rows':<15} {'Status':<15} {'Sufficient':<15}")
        report.append("-" * 80)

        for key in ["bhavcopy", "vix", "fii_dii", "global_markets"]:
            if key in self.results:
                r = self.results[key]
                name = r.get("name", key)
                rows = r.get("rows", 0)
                status = r.get("status", "UNKNOWN")
                sufficient = "✅ YES" if r.get("sufficient", False) else "❌ NO"
                report.append(f"{name:<25} {rows:<15,} {status:<15} {sufficient:<15}")

        report.append("-" * 80)
        report.append("")

        # ML Requirements
        if "ml_requirements" in self.results:
            ml = self.results["ml_requirements"]
            report.append("ML MODEL REQUIREMENTS")
            report.append("-" * 80)
            report.append(f"Total rows: {ml['total_rows']:,}")
            report.append(f"Usable sequences: {ml['usable_sequences']:,}")
            report.append(f"Number of features: {ml['num_features']}")
            report.append(f"TFT parameters: {ml['tft_parameters']:,}")
            report.append("")
            report.append(f"Feature-based requirement: {ml['min_sequences_features']:,} sequences")
            report.append(f"  Ratio: {ml['feature_requirement_ratio']:.2f}x ({'✅ PASS' if ml['feature_based_sufficient'] else '❌ FAIL'})")
            report.append("")
            report.append(f"Parameter-based requirement: {ml['min_sequences_params']:,} sequences")
            report.append(f"  Ratio: {ml['parameter_requirement_ratio']:.3f}x ({'✅ PASS' if ml['parameter_based_sufficient'] else '⚠️ MARGINAL'})")
            report.append("")
            report.append(f"Overall Assessment: {ml['overall_assessment']}")
            report.append("-" * 80)
            report.append("")

        # Errors and Warnings
        if self.errors:
            report.append("CRITICAL ERRORS")
            report.append("-" * 80)
            for error in self.errors:
                report.append(f"❌ {error}")
            report.append("")

        if self.warnings:
            report.append("WARNINGS")
            report.append("-" * 80)
            for warning in self.warnings:
                report.append(f"⚠️  {warning}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)

        # Check FII/DII
        if "fii_dii" in self.results and self.results["fii_dii"]["rows"] < 100:
            report.append("🔴 CRITICAL: Download FII/DII data immediately")
            report.append("   Command: python data_downloader.py --fii-dii --start-date 2020-01-01")
            report.append("")

        # Check global markets
        if "global_markets" in self.results and self.results["global_markets"]["rows"] < 50000:
            report.append("🟡 HIGH PRIORITY: Extend global markets data")
            report.append("   Command: python market_data_extended.py --start-date 2020-01-01")
            report.append("")

        # Check bhavcopy
        if "bhavcopy" in self.results and self.results["bhavcopy"]["days_coverage"] < 504:
            report.append("🟡 HIGH PRIORITY: Download more bhavcopy data")
            report.append("   Command: python data_downloader.py --bhavcopy --start-date 2020-01-01")
            report.append("")

        # General recommendation
        if not self.errors and not self.warnings:
            report.append("✅ All data sources meet minimum requirements")
            report.append("✅ Ready to train production model")
        elif self.errors:
            report.append("❌ Critical data gaps detected")
            report.append("❌ NOT recommended to train production model until gaps are filled")
        else:
            report.append("⚠️  Data is marginally sufficient")
            report.append("⚠️  Can train baseline model, but fill gaps for production")

        report.append("=" * 80)

        return "\n".join(report)

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        self.log("Starting data sufficiency validation...", "INFO")
        print()

        # Check each data source
        self.check_bhavcopy_data()
        print()

        self.check_vix_data()
        print()

        self.check_fii_dii_data()
        print()

        self.check_global_markets_data()
        print()

        # Calculate ML requirements
        self.calculate_ml_requirements()
        print()

        # Generate and print report
        report = self.generate_report()
        print(report)

        # Return True if no critical errors
        return len(self.errors) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate data sufficiency for ML/NN training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Export report to file"
    )

    args = parser.parse_args()

    # Create validator
    validator = DataSufficiencyValidator(verbose=args.verbose)

    # Run checks
    success = validator.run_all_checks()

    # Export report if requested
    if args.export_report:
        report = validator.generate_report()
        output_file = Path("data_sufficiency_report.txt")
        output_file.write_text(report)
        print(f"\n📄 Report exported to: {output_file}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
