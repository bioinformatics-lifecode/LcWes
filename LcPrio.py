#!/usr/bin/env python3
"""
Clinical Variant Prioritization for WES Data (CVPW)
---------------------------------------------------

A comprehensive variant prioritization script for clinical diagnostics
that ranks variants based on pathogenicity, clinical significance,
inheritance patterns, and in silico predictions while preserving the original format.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import re
import json
import logging
from collections import defaultdict, Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('CVPW')

# Constants for variant classification and scoring
IMPACT_CATEGORIES = {
    'HIGH': ['frameshift_variant', 'stop_gained', 'stop_lost', 'start_lost', 
            'splice_acceptor_variant', 'splice_donor_variant', 'transcript_ablation'],
    'MODERATE': ['missense_variant', 'inframe_insertion', 'inframe_deletion', 
                'protein_altering_variant', 'regulatory_region_ablation'],
    'LOW': ['synonymous_variant', 'splice_region_variant', '5_prime_UTR_variant', 
           '3_prime_UTR_variant', 'upstream_gene_variant', 'downstream_gene_variant'],
    'MODIFIER': ['intron_variant', 'intergenic_variant', 'non_coding_transcript_variant']
}

# ACMG rule weights
ACMG_RULE_WEIGHTS = {
    # Very Strong Evidence
    'PVS1': 30,
    
    # Strong Evidence
    'PS1': 10, 'PS2': 10, 'PS3': 10, 'PS4': 10,
    'BA1': -7,
    
    # Moderate Evidence
    'PM1': 6, 'PM2': 6, 'PM3': 6, 'PM4': 6, 'PM5': 6, 'PM6': 6,
    'BS1': -5, 'BS2': -5, 'BS3': -5, 'BS4': -5,
    
    # Supporting Evidence
    'PP1': 3, 'PP2': 3, 'PP3': 3, 'PP4': 3, 'PP5': 3,
    'BP1': -2, 'BP2': -2, 'BP3': -2, 'BP4': -2, 'BP5': -2, 'BP6': -2, 'BP7': -2
}

class VariantPrioritization:
    """Main class for variant prioritization"""
    
    def __init__(self, args):
        """Initialize with command line arguments"""
        self.args = args
        self.input_file = args.input
        self.output_file = args.output
        self.top_n = args.top
        self.gene_list_file = args.genes
        self.phenotype_file = args.phenotype
        self.inheritance_file = args.inheritance
        self.include_benign = args.include_benign
        self.min_cadd = args.min_cadd
        self.max_gnomad = args.max_gnomad
        self.output_format = args.format
        
        # Column name mapping
        self.column_map = {}
        
        # Load supporting data
        self.genes_of_interest = self._load_gene_list()
        self.phenotype_terms = self._load_phenotype_terms()
        self.inheritance_patterns = self._load_inheritance_patterns()
        
        # Stats tracking
        self.stats = {
            'total_variants': 0,
            'pathogenic': 0,
            'likely_pathogenic': 0,
            'vus': 0,
            'likely_benign': 0,
            'benign': 0,
            'high_impact': 0,
            'moderate_impact': 0,
            'low_impact': 0
        }
    
    def _load_gene_list(self):
        """Load list of genes of interest from file"""
        if not self.gene_list_file or not os.path.exists(self.gene_list_file):
            return set()
            
        try:
            with open(self.gene_list_file, 'r') as f:
                return {line.strip() for line in f if line.strip() and not line.startswith('#')}
        except Exception as e:
            logger.warning(f"Error loading gene list: {e}")
            return set()
    
    def _load_phenotype_terms(self):
        """Load phenotype terms for matching"""
        if not self.phenotype_file or not os.path.exists(self.phenotype_file):
            return []
            
        try:
            with open(self.phenotype_file, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            logger.warning(f"Error loading phenotype terms: {e}")
            return []
    
    def _load_inheritance_patterns(self):
        """Load gene-specific inheritance patterns"""
        if not self.inheritance_file or not os.path.exists(self.inheritance_file):
            return {}
            
        try:
            patterns = {}
            with open(self.inheritance_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gene, inheritance = parts[0], parts[1]
                            patterns[gene] = inheritance
            return patterns
        except Exception as e:
            logger.warning(f"Error loading inheritance patterns: {e}")
            return {}
    
    def _map_columns(self, df):
        """Map common column naming variations to standardized names"""
        # Log all column names for debugging
        logger.info(f"Available columns: {', '.join(df.columns)}")
        
        # Create mapping for important columns with potential name variations
        column_patterns = {
            'ACMG': ['ACMG', 'acmg', 'ACMG Classification'],
            'ACMG_Rules': ['ACMG_Rules', 'acmg_rules', 'ACMG Rules'],
            'clinvar': ['clinvar: Clinvar ', 'clinvar: Clinvar', 'clinvar:', 'Clinvar', 'clinvar'],
            'CLNSIGCONF': ['CLNSIGCONF', 'clnsigconf', 'ClinVar_conflicting'],
            'CADD_phred': ['CADD_phred', 'CADD_PHRED', 'CADD Phred'],
            'SIFT_score': ['SIFT_score', 'SIFT Score', 'SIFT_Score'],
            'gnomAD_freq': ['Freq_gnomAD_genome_ALL', 'gnomAD_AF', 'gnomAD_freq'],
            'esp_freq': ['Freq_esp6500siv2_all', 'ESP_freq', 'ESP6500_freq'],
            'g1000_freq': ['Freq_1000g2015aug_all', '1000g_freq', '1000G_freq'],
            'pop_freqs': ['Freq_gnomAD_genome_POPs', 'gnomAD_pops', 'gnomAD_POPs'],
            'GERP': ['GERP++_RS', 'GERP_RS', 'GERP'],
            'phyloP': ['phyloP46way_placental', 'phyloP', 'PhyloP'],
            'ADA_score': ['dbscSNV_ADA_SCORE', 'ADA_SCORE', 'ada_score'],
            'RF_score': ['dbscSNV_RF_SCORE', 'RF_SCORE', 'rf_score'],
            'MetaSVM': ['MetaSVM_score', 'metaSVM', 'MetaSVM Score'],
            'gene': ['ANN[0].GENE', 'Gene', 'RefGene', 'Ref.Gene'],
            'effect': ['ANN[0].EFFECT', 'Effect', 'ExonicFunc', 'ExonicFunc.refGene'],
            'impact': ['ANN[0].IMPACT', 'Impact', 'Func.refGene'],
            'hgvs_p': ['ANN[0].HGVS_P', 'HGVS_P', 'AAChange'],
            'hgvs_c': ['ANN[0].HGVS_C', 'HGVS_C', 'GeneDetail'],
            'depth': ['DP', 'depth', 'Coverage'],
            'af': ['AF', 'alt_freq', 'VAF'],
            'ad': ['GEN[0].AD', 'AD', 'Allele Depth'],
            'OMIM': ['OMIM', 'omim', 'OMIM_Gene'],
            'Orpha': ['Orpha', 'orpha', 'OrphaNumber', 'Orphanet'],
            'origin': ['origin', 'Origin', 'Inheritance'],
            'filter': ['FILTER', 'filter', 'Filter']
        }
        
        # Create the mapping
        self.column_map = {}
        for standard, variations in column_patterns.items():
            for var in variations:
                if var in df.columns:
                    self.column_map[standard] = var
                    break
        
        # Log the mapping
        logger.info(f"Column mapping: {self.column_map}")
        
        # Warn about missing important columns
        important_columns = ['ACMG', 'gene', 'effect', 'impact']
        for col in important_columns:
            if col not in self.column_map:
                logger.warning(f"Important column '{col}' not found in the input file")
        
        return self.column_map
    
    def _get_col(self, name, row, default='.'):
        """Safely get a column value using the column mapping"""
        if name in self.column_map and self.column_map[name] in row:
            val = row[self.column_map[name]]
            return default if pd.isna(val) else val
        return default
    
    def _get_col_as_str(self, name, row, default='.'):
        """Get a column value as a string, handling different data types"""
        val = self._get_col(name, row, default)
        if pd.isna(val):
            return default
        return str(val)
    
    def _parse_clnsigconf(self, clnsigconf_val):
        """Parse CLNSIGCONF field and return score based on pathogenic classifications"""
        if clnsigconf_val == '.':
            return 0
        
        score = 0
        
        # Split by | to get individual classifications
        classifications = clnsigconf_val.split('|')
        
        for classification in classifications:
            # Use regex to extract classification name and count
            # Pattern: Classification_name(count)
            match = re.match(r'([^(]+)\((\d+)\)', classification.strip())
            if match:
                cls_name = match.group(1).strip().lower()
                count = int(match.group(2))
                
                # Add score for classifications containing pathogenic terms
                # This handles cases like "Pathogenic", "Pathogenic/Likely_pathogenic", "Pathogenic|association", etc.
                if 'pathogenic' in cls_name and 'likely_pathogenic' not in cls_name:
                    # Pure pathogenic (including combinations like Pathogenic/Likely_risk_allele, Pathogenic|association)
                    score += count * 200  # High weight for pathogenic
                elif 'likely_pathogenic' in cls_name:
                    # Likely pathogenic (including combinations like Likely_pathogenic|risk_factor)
                    score += count * 150  # Moderate weight for likely pathogenic
        
        return score
    
    def run(self):
        """Run the complete prioritization process"""
        logger.info(f"Processing file: {self.input_file}")
        
        # Read and preprocess data
        df = self.read_input_file()
        if df is None or df.empty:
            return False
        
        # Save original data for later use
        original_df = df.copy()
        
        # Create column mapping
        self._map_columns(df)
        
        # Apply filtration based on command line arguments
        df = self.apply_prefilters(df)
        
        # Calculate priority scores without modifying the DataFrame
        priority_scores = []
        classifications = []
        component_scores = {}  # Store component scores for each variant
        
        for idx, row in df.iterrows():
            # Calculate all component scores
            clinical_score = self.calculate_clinical_score(row)
            impact_score = self.calculate_impact_score(row)
            frequency_score = self.calculate_frequency_score(row)
            prediction_score = self.calculate_prediction_score(row)
            acmg_rule_score = self.calculate_acmg_rule_score(row)
            conservation_score = self.calculate_conservation_score(row)
            inheritance_score = self.calculate_inheritance_score(row)
            phenotype_match_score = self.calculate_phenotype_score(row)
            quality_score = self.calculate_quality_score(row)
            
            # Store component scores for this variant
            component_scores[idx] = {
                'ACMG/Clinical': clinical_score * 5,
                'Impact': impact_score * 3,
                'Frequency': frequency_score * 2,
                'Prediction': prediction_score * 2,
                'ACMG_Rules': acmg_rule_score * 4,
                'Conservation': conservation_score * 1,
                'Inheritance': inheritance_score * 2,
                'Phenotype': phenotype_match_score * 3,
                'Quality': quality_score * 1
            }
            
            # Calculate final priority score
            priority_score = (
                clinical_score * 5 +
                impact_score * 3 +
                frequency_score * 2 +
                prediction_score * 2 +
                acmg_rule_score * 4 +
                conservation_score * 1 +
                inheritance_score * 2 +
                phenotype_match_score * 3 +
                quality_score * 1
            )
            
            priority_scores.append((idx, priority_score))
            classifications.append((idx, self.get_variant_classification(row)))
        
        # Create a mapping of original index to priority score and classification
        priority_score_map = dict(priority_scores)
        classification_map = dict(classifications)
        
        # Count classifications for statistics
        classification_counts = Counter(classification_map.values())
        self.stats['pathogenic'] = classification_counts.get('Pathogenic', 0)
        self.stats['likely_pathogenic'] = classification_counts.get('Likely Pathogenic', 0)
        self.stats['vus'] = classification_counts.get('VUS', 0) + classification_counts.get('Uncertain', 0) + classification_counts.get('Unknown', 0)
        self.stats['likely_benign'] = classification_counts.get('Likely Benign', 0)
        self.stats['benign'] = classification_counts.get('Benign', 0)
        
        # Sort based on priority scores
        sorted_indices = sorted(priority_score_map.keys(), key=lambda x: priority_score_map[x], reverse=True)
        
        # Filter for top variants if specified
        if self.top_n > 0 and len(sorted_indices) > self.top_n:
            logger.info(f"Selecting top {self.top_n} variants")
            sorted_indices = sorted_indices[:self.top_n]
        
        # Get the subset of original DataFrame with sorted indices
        df_sorted = original_df.loc[sorted_indices].copy()
        
        # Write output
        self.write_output(df_sorted, sorted_indices, priority_score_map, classification_map, component_scores)
        
        # Report statistics
        self.report_stats()
        
        logger.info("Prioritization complete.")
        return True
    
    def read_input_file(self):
        """Read and preprocess input variant file"""
        try:
            # Read the file with all columns as strings to preserve original format
            df = pd.read_csv(self.input_file, sep='\t', dtype=str, na_filter=False)
            
            self.stats['total_variants'] = len(df)
            logger.info(f"Read {len(df)} variants from {self.input_file}")
            
            return df
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            return None
    
    def apply_prefilters(self, df):
        """Apply basic filtering based on command line arguments"""
        original_count = len(df)
        
        # Filter by CADD score if specified
        if self.min_cadd > 0 and 'CADD_phred' in self.column_map:
            cadd_col = self.column_map['CADD_phred']
            try:
                # Convert CADD to numeric, replacing non-convertible values with NaN
                cadd_scores = pd.to_numeric(df[cadd_col], errors='coerce')
                df = df[((cadd_scores >= self.min_cadd) | (cadd_scores.isna()))]
                logger.info(f"Filtered variants by CADD >= {self.min_cadd}: {len(df)} remaining")
            except Exception as e:
                logger.warning(f"Error filtering by CADD: {e}")
        
        # Filter by gnomAD frequency if specified
        if self.max_gnomad < 1.0:
            gnomad_col = self.column_map.get('gnomAD_freq', None)
            if gnomad_col:
                try:
                    # Convert frequency to numeric, handling missing/invalid values
                    gnomad_freq = pd.to_numeric(df[gnomad_col].replace('.', np.nan), errors='coerce')
                    df = df[gnomad_freq.isna() | (gnomad_freq <= self.max_gnomad)]
                    logger.info(f"Filtered variants by gnomAD frequency <= {self.max_gnomad}: {len(df)} remaining")
                except Exception as e:
                    logger.warning(f"Error filtering by gnomAD frequency: {e}")
        
        # Exclude benign variants if specified
        if not self.include_benign and 'ACMG' in self.column_map:
            acmg_col = self.column_map['ACMG']
            try:
                # Keep variants that aren't clearly benign
                benign_mask = df[acmg_col].astype(str).str.contains('enign', case=False, na=False) & ~df[acmg_col].astype(str).str.contains('ikely', case=False, na=False)
                df = df[~benign_mask]
                logger.info(f"Excluded benign variants: {len(df)} remaining")
            except Exception as e:
                logger.warning(f"Error excluding benign variants: {e}")
        
        # Filter for genes of interest if provided
        if self.genes_of_interest and 'gene' in self.column_map:
            gene_col = self.column_map['gene']
            try:
                df = df[df[gene_col].isin(self.genes_of_interest)]
                logger.info(f"Filtered for {len(self.genes_of_interest)} genes of interest: {len(df)} remaining")
            except Exception as e:
                logger.warning(f"Error filtering for genes of interest: {e}")
        
        logger.info(f"Applied prefilters: {original_count - len(df)} variants removed, {len(df)} variants remaining")
        return df
    
    def calculate_clinical_score(self, row):
        """Score based on clinical significance from ACMG and ClinVar"""
        score = 0
        
        # ACMG Classification
        acmg_val = self._get_col_as_str('ACMG', row)
        if acmg_val != '.':
            acmg_class = acmg_val.lower()
            if 'pathogenic' in acmg_class and 'likely' not in acmg_class:
                score += 2000  # Pathogenic
            elif 'likely pathogenic' in acmg_class:
                score += 1800   # Likely pathogenic
            elif 'uncertain significance' in acmg_class:
                score += 600   # VUS
            elif 'likely benign' in acmg_class:
                score += 200   # Likely benign
            elif 'benign' in acmg_class:
                score += 100   # Benign
        
        # ClinVar annotation - comprehensive scoring with stronger impact for pathogenic variants
        clinvar_val = self._get_col_as_str('clinvar', row)
        if clinvar_val != '.':
            clinvar = clinvar_val.lower()
            
            # Pure Pathogenic variants - highest priority
            if ('pathogenic' in clinvar and 'likely_pathogenic' not in clinvar and 
                'benign' not in clinvar and 'conflicting' not in clinvar and 
                'uncertain' not in clinvar):
                score += 1800  # Pure pathogenic - highest score
                
            # Pathogenic with low penetrance
            elif 'pathogenic\\x2c_low_penetrance' in clinvar and 'benign' not in clinvar:
                score += 1700  # Pathogenic with low penetrance
                
            # Pathogenic/Likely_pathogenic combinations
            elif (('pathogenic/likely_pathogenic' in clinvar or 'likely_pathogenic/pathogenic' in clinvar) and 
                  'benign' not in clinvar):
                score += 1600  # Pathogenic/Likely_pathogenic
                
            # Pathogenic with risk allele
            elif 'pathogenic/likely_risk_allele' in clinvar and 'benign' not in clinvar:
                score += 1500  # Pathogenic with risk allele
                
            # Other pathogenic combinations
            elif ('pathogenic/' in clinvar or '/pathogenic' in clinvar) and 'benign' not in clinvar:
                score += 1400  # Other pathogenic combinations
                
            # Standalone Likely_pathogenic
            elif 'likely_pathogenic' in clinvar and 'pathogenic/' not in clinvar and '/pathogenic' not in clinvar and 'benign' not in clinvar:
                score += 1400  # Likely_pathogenic
                
            # Likely risk allele
            elif 'likely_risk_allele' in clinvar and 'uncertain' not in clinvar:
                score += 500  # Likely risk allele
                
            # Conflicting classifications involving pathogenic
            elif 'conflicting_classifications_of_pathogenicity' in clinvar:
                score += 400  # Conflicting with pathogenic involved
            
            # VUS - Uncertain significance
            elif 'uncertain_significance' in clinvar:
                score += 300  # Uncertain significance
                
            # Uncertain risk allele
            elif 'uncertain_risk_allele' in clinvar:
                score += 250  # Uncertain risk allele
            
            # Likely benign
            elif 'likely_benign' in clinvar and 'pathogenic' not in clinvar and 'conflicting' not in clinvar:
                score += 100  # Likely benign
            
            # Benign
            elif 'benign' in clinvar and 'likely' not in clinvar and 'pathogenic' not in clinvar and 'conflicting' not in clinvar:
                score += 50   # Benign
            
            # Additional modifiers - these add to any of the above scores
            if 'affects' in clinvar:
                score += 120  # Affects function (increased from 100)
            if 'risk_factor' in clinvar:
                score += 180  # Risk factor (increased from 150)
            if 'association' in clinvar:
                score += 100  # Association (new category)
            if 'protective' in clinvar:
                score += 120  # Protective factor (increased from 100)
            if 'drug_response' in clinvar:
                score += 80   # Drug response (increased from 50)
            if 'confers_sensitivity' in clinvar:
                score += 100  # Confers sensitivity (new category)
            if 'low_penetrance' in clinvar and '\\x2c_low_penetrance' not in clinvar:
                score += 70   # Low penetrance (increased from 50)
        
        # Add CLNSIGCONF scoring for conflicting pathogenic classifications
        clnsigconf_val = self._get_col_as_str('CLNSIGCONF', row)
        clnsigconf_score = self._parse_clnsigconf(clnsigconf_val)
        score += clnsigconf_score
        
        return score
    
    def calculate_impact_score(self, row):
        """Score based on variant impact and effect"""
        score = 0
        
        # Check variant effect
        effect_val = self._get_col_as_str('effect', row)
        if effect_val != '.':
            effect = effect_val.lower()
            
            # High impact variants
            if any(x in effect for x in ['frameshift', 'stop_gained', 'stop_lost', 'start_lost', 
                                        'splice_donor', 'splice_acceptor']):
                score += 500
                self.stats['high_impact'] += 1
            
            # Moderate impact variants
            elif any(x in effect for x in ['missense', 'inframe_insertion', 'inframe_deletion',
                                          'protein_altering', 'splice_region']):
                score += 300
                self.stats['moderate_impact'] += 1
            
            # Low impact coding variants
            elif any(x in effect for x in ['synonymous', 'stop_retained', 'start_retained']):
                score += 150
                self.stats['low_impact'] += 1
            
            # Non-coding variants
            elif any(x in effect for x in ['utr', 'intron', 'upstream', 'downstream', 
                                         'intergenic', 'non_coding']):
                score += 50
        
        # Check impact from SnpEff annotation
        impact_val = self._get_col_as_str('impact', row)
        if impact_val != '.':
            impact = impact_val.upper()
            if impact == 'HIGH':
                score += 300
            elif impact == 'MODERATE':
                score += 200
            elif impact == 'LOW':
                score += 100
            elif impact == 'MODIFIER':
                score += 50
        
        return score
    
    def calculate_frequency_score(self, row):
        """Score based on population frequency (rarer = higher score)"""
        score = 0
        
        # Check various frequency fields with priority order
        for freq_field in ['gnomAD_freq', 'esp_freq', 'g1000_freq']:
            freq_val = self._get_col(freq_field, row)
            if freq_val != '.':
                try:
                    freq = float(freq_val)
                    if freq == 0:  # Novel variant
                        score += 500
                    elif freq < 0.0001:  # Extremely rare
                        score += 400
                    elif freq < 0.001:  # Very rare
                        score += 300
                    elif freq < 0.01:   # Rare
                        score += 200
                    elif freq < 0.05:   # Uncommon
                        score += 100
                    else:               # Common
                        score += 0
                    break  # Only use the first valid frequency found
                except (ValueError, TypeError):
                    continue
        
        # Check for presence in population-specific frequencies
        pop_freqs_val = self._get_col_as_str('pop_freqs', row)
        if pop_freqs_val != '.':
            try:
                # Check if variant is more common in specific populations
                pops = pop_freqs_val
                if pops != '.':
                    # If variant is much more common in a specific population, reduce score slightly
                    pop_freqs = re.findall(r'([A-Z]+):([0-9.]+)', pops)
                    if pop_freqs:
                        max_pop_freq = max(float(freq) for _, freq in pop_freqs)
                        if max_pop_freq > 0.05:  # Common in at least one population
                            score -= 50
            except:
                pass
                
        return score
    
    def calculate_prediction_score(self, row):
        """Score based on in silico prediction tools"""
        score = 0
        
        # CADD score (higher is more deleterious)
        cadd_val = self._get_col('CADD_phred', row)
        if cadd_val != '.':
            try:
                cadd = float(cadd_val)
                if cadd > 30:      # Extremely deleterious
                    score += 300
                elif cadd > 25:    # Very highly deleterious 
                    score += 250
                elif cadd > 20:    # Highly deleterious
                    score += 200
                elif cadd > 15:    # Moderately deleterious
                    score += 150
                elif cadd > 10:    # Possibly deleterious
                    score += 100
                else:              # Likely benign
                    score += 0
            except (ValueError, TypeError):
                pass
        
        # SIFT score (lower is more deleterious)
        sift_val = self._get_col('SIFT_score', row)
        if sift_val != '.':
            try:
                sift = float(sift_val)
                if sift < 0.05:    # Deleterious
                    score += 200
                elif sift < 0.1:   # Possibly deleterious
                    score += 100
                elif sift < 0.2:   # Borderline
                    score += 50
                else:              # Tolerated
                    score += 0
            except (ValueError, TypeError):
                pass
        
        # MetaSVM score (higher is more deleterious)
        metasvm_val = self._get_col('MetaSVM', row)
        if metasvm_val != '.':
            try:
                metasvm = float(metasvm_val)
                if metasvm > 0.5:   # Predicted deleterious
                    score += 150
            except (ValueError, TypeError):
                pass
        
        # dbscSNV scores for splicing
        for splicing_score in ['ADA_score', 'RF_score']:
            splicing_val = self._get_col(splicing_score, row)
            if splicing_val != '.':
                try:
                    value = float(splicing_val)
                    if value > 0.8:     # Strong prediction of splicing effect
                        score += 150
                    elif value > 0.6:   # Moderate prediction
                        score += 75
                except (ValueError, TypeError):
                    pass
        
        return score
    
    def calculate_acmg_rule_score(self, row):
        """Calculate score based on ACMG rules"""
        acmg_rules_val = self._get_col_as_str('ACMG_Rules', row)
        if acmg_rules_val == '.':
            return 0
        
        # Split the rules by comma
        rules = acmg_rules_val.split(',')
        
        # Calculate total score
        total_score = sum(ACMG_RULE_WEIGHTS.get(rule.strip(), 0) for rule in rules)
        
        # Scale the score for better integration with other scores
        return total_score * 20
    
    def calculate_conservation_score(self, row):
        """Calculate conservation score from GERP and phyloP"""
        score = 0
        
        # GERP score (higher is more conserved)
        gerp_val = self._get_col('GERP', row)
        if gerp_val != '.':
            try:
                gerp = float(gerp_val)
                if gerp > 5:      # Extremely conserved
                    score += 200
                elif gerp > 4:    # Highly conserved
                    score += 150
                elif gerp > 2:    # Moderately conserved
                    score += 100
                elif gerp > 0:    # Slightly conserved
                    score += 50
            except (ValueError, TypeError):
                pass
        
        # phyloP score (higher is more conserved)
        phylop_val = self._get_col('phyloP', row)
        if phylop_val != '.':
            try:
                phylop = float(phylop_val)
                if phylop > 3:     # Extremely conserved
                    score += 150
                elif phylop > 2:   # Highly conserved
                    score += 100
                elif phylop > 1:   # Moderately conserved
                    score += 75
                elif phylop > 0:   # Slightly conserved
                    score += 50
            except (ValueError, TypeError):
                pass
        
        return score
    
    def calculate_inheritance_score(self, row):
        """Score based on inheritance patterns and genotype"""
        score = 0
        
        # Check for origin field information
        origin_val = self._get_col_as_str('origin', row)
        if origin_val != '.':
            origin = origin_val.lower()
            
            # De novo variants get high priority
            if 'de novo' in origin or 'denovo' in origin:
                score += 500
                
            # Compound heterozygous variants for recessive conditions
            elif 'compound' in origin and 'heterozygous' in origin:
                score += 300
                
            # Homozygous variants in recessive conditions
            elif 'homozygous' in origin:
                score += 300
                
            # Hemizygous variants in X-linked conditions
            elif 'hemizygous' in origin or 'hemizygote' in origin:
                score += 300
                
            # Potentially interesting inheritance patterns
            elif any(x in origin for x in ['x-linked', 'dominant', 'recessive']):
                score += 200
        
        # Check for homozygous variants based on genotype AD field
        ad_val = self._get_col_as_str('ad', row)
        if ad_val != '.' and ',' in ad_val:
            try:
                ad_parts = ad_val.split(',')
                ref_depth = int(ad_parts[0])
                alt_depth = int(ad_parts[1]) if len(ad_parts) > 1 else 0
                
                # Calculate VAF (Variant Allele Frequency)
                total_depth = ref_depth + alt_depth
                if total_depth > 0:
                    vaf = alt_depth / total_depth
                    
                    # Homozygous if VAF > 0.8
                    if vaf > 0.8:
                        score += 200
                    # Heterozygous with good balance
                    elif 0.3 <= vaf <= 0.7:
                        score += 100
            except (ValueError, IndexError, AttributeError):
                # Handle various exceptions when parsing AD field
                pass
        
        # Check gene-specific inheritance patterns if available
        gene_val = self._get_col_as_str('gene', row)
        if gene_val != '.':
            gene = gene_val
            if gene in self.inheritance_patterns:
                inheritance = self.inheritance_patterns[gene].lower()
                
                # Higher score for genes with known inheritance patterns
                score += 150
                
                # Additional score for X-linked genes
                if 'x-linked' in inheritance:
                    score += 100
        
        return score
    
    def calculate_phenotype_score(self, row):
        """Score based on phenotype matches"""
        score = 0
        
        # Check if we have phenotype terms to match
        if not self.phenotype_terms:
            return 0
        
        # Check OMIM description for phenotype matches
        omim_val = self._get_col_as_str('OMIM', row)
        if omim_val != '.':
            omim_text = str(omim_val).lower()
            for term in self.phenotype_terms:
                if term.lower() in omim_text:
                    score += 100  # Add score per matching phenotype term
        
        # Check Orphanet description for phenotype matches
        orpha_val = self._get_col_as_str('Orpha', row)
        if orpha_val != '.':
            orpha_text = str(orpha_val).lower()
            for term in self.phenotype_terms:
                if term.lower() in orpha_text:
                    score += 100  # Add score per matching phenotype term
        
        # Boost for genes in our genes of interest list
        gene_val = self._get_col_as_str('gene', row)
        if gene_val != '.' and gene_val in self.genes_of_interest:
            score += 200
        
        return score
    
    def calculate_quality_score(self, row):
        """Score based on quality metrics"""
        score = 0
        
        # Depth of coverage
        depth_val = self._get_col('depth', row)
        if depth_val != '.':
            try:
                depth = float(depth_val)
                if depth >= 50:      # Excellent coverage
                    score += 150
                elif depth >= 30:    # Good coverage
                    score += 100
                elif depth >= 20:    # Acceptable coverage
                    score += 50
                elif depth >= 10:    # Minimal coverage
                    score += 0
                else:                # Poor coverage
                    score -= 100     # Penalize low coverage
            except (ValueError, TypeError):
                pass
        
        # Allele frequency in sample
        af_val = self._get_col('af', row)
        if af_val != '.':
            try:
                af = float(af_val)
                if af >= 0.3:  # Strong variant signal
                    score += 100
                elif af >= 0.2:
                    score += 50
                elif af < 0.1:  # Potential sequencing artifact
                    score -= 50
            except (ValueError, TypeError):
                pass
        
        # Filter status
        filter_val = self._get_col_as_str('filter', row)
        if filter_val != '.':
            if filter_val == 'PASS':
                score += 100  # Passed filters
            else:
                score -= 200  # Failed filters
        
        return score
    
    def get_variant_classification(self, row):
        """Get standardized classification for a variant"""
        # First check ACMG
        acmg_val = self._get_col_as_str('ACMG', row)
        if acmg_val != '.':
            acmg = acmg_val.lower()
            if 'pathogenic' in acmg and 'likely' not in acmg:
                return 'Pathogenic'
            elif 'likely pathogenic' in acmg:
                return 'Likely Pathogenic'
            elif 'uncertain significance' in acmg:
                return 'VUS'
            elif 'likely benign' in acmg:
                return 'Likely Benign'
            elif 'benign' in acmg and 'likely' not in acmg:
                return 'Benign'
        
        # If no ACMG, check ClinVar
        clinvar_val = self._get_col_as_str('clinvar', row)
        if clinvar_val != '.':
            clinvar = clinvar_val.lower()
            if 'pathogenic' in clinvar and 'likely' not in clinvar and 'benign' not in clinvar:
                return 'Pathogenic'
            elif 'likely_pathogenic' in clinvar and 'benign' not in clinvar:
                return 'Likely Pathogenic'
            elif 'uncertain_significance' in clinvar:
                return 'VUS'
            elif 'likely_benign' in clinvar and 'pathogenic' not in clinvar:
                return 'Likely Benign'
            elif 'benign' in clinvar and 'likely' not in clinvar and 'pathogenic' not in clinvar:
                return 'Benign'
            elif 'conflicting' in clinvar:
                return 'Conflicting'
        
        # Default to unknown
        return 'Unknown'
    
    def format_component_scores(self, scores_dict):
        """Format component scores as a readable string"""
        components = []
        for name, score in scores_dict.items():
            if score > 0:
                components.append(f"{name}: {score}")
        return ', '.join(components)
    
    def write_output(self, df_sorted, sorted_indices, priority_score_map, classification_map, component_scores):
        """Write prioritized variants to output file, preserving original format"""
        logger.info(f"Writing {len(df_sorted)} prioritized variants to {self.output_file}")
        
        # Add the priority scores and component scores to the output DataFrame
        df_sorted['PriorityScore'] = [priority_score_map[idx] for idx in sorted_indices]
        df_sorted['ScoreComponents'] = [self.format_component_scores(component_scores[idx]) for idx in sorted_indices]
        
        # Basic TSV output - with additional columns
        df_sorted.to_csv(self.output_file, sep='\t', index=False)
        
        # Write summary file
        self.write_summary(df_sorted, priority_score_map, classification_map)
        
        # Write additional formats if requested
        if self.output_format == 'excel':
            excel_file = os.path.splitext(self.output_file)[0] + '.xlsx'
            try:
                df_sorted.to_excel(excel_file, index=False)
                logger.info(f"Excel output written to {excel_file}")
            except Exception as e:
                logger.error(f"Error writing Excel output: {e}")
        
        elif self.output_format == 'json':
            json_file = os.path.splitext(self.output_file)[0] + '.json'
            try:
                # Convert DataFrame to JSON
                df_json = df_sorted.to_dict(orient='records')
                
                # Add priority scores and classifications
                for i, record in enumerate(df_json):
                    idx = sorted_indices[i]
                    record['priority_score'] = priority_score_map[idx]
                    record['classification'] = classification_map[idx]
                    record['score_components'] = component_scores[idx]
                
                result = {
                    'metadata': {
                        'input_file': self.input_file,
                        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_variants': len(df_sorted),
                        'stats': self.stats
                    },
                    'variants': df_json
                }
                
                with open(json_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                logger.info(f"JSON output written to {json_file}")
            except Exception as e:
                logger.error(f"Error writing JSON output: {e}")
    
    def write_summary(self, df_sorted, priority_score_map, classification_map):
        """Write summary of prioritized variants"""
        summary_file = os.path.splitext(self.output_file)[0] + '.summary.txt'
        
        # Get classification distribution
        classification_counts = Counter(classification_map.values())
        
        # Get gene distribution
        gene_col = self.column_map.get('gene', None)
        gene_counts = Counter()
        if gene_col:
            gene_counts = Counter(df_sorted[gene_col].values)
        
        # Get impact distribution
        impact_col = self.column_map.get('impact', None)
        impact_counts = Counter()
        if impact_col:
            impact_counts = Counter(df_sorted[impact_col].values)
        
        # Write summary file
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Variant Prioritization Summary\n")
                f.write(f"==============================\n\n")
                f.write(f"Input file: {self.input_file}\n")
                f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Variant Statistics:\n")
                f.write(f"------------------\n")
                f.write(f"Total variants processed: {self.stats['total_variants']}\n")
                f.write(f"Variants after filtering: {len(df_sorted)}\n\n")
                
                f.write(f"Classification Distribution:\n")
                for cls, count in sorted(classification_counts.items(), key=lambda x: (x[0] != 'Pathogenic', x[0] != 'Likely Pathogenic', x[0])):
                    f.write(f"  {cls}: {count}\n")
                f.write("\n")
                
                if impact_counts:
                    f.write(f"Impact Distribution:\n")
                    for impact, count in sorted(impact_counts.items(), key=lambda x: (str(x[0]) != 'HIGH', str(x[0]) != 'MODERATE', x[0])):
                        f.write(f"  {impact}: {count}\n")
                    f.write("\n")
                
                if gene_counts:
                    f.write(f"Top Genes:\n")
                    for gene, count in sorted(gene_counts.most_common(10)):
                        if gene != '.':
                            f.write(f"  {gene}: {count}\n")
                
            logger.info(f"Summary written to {summary_file}")
        except Exception as e:
            logger.error(f"Error writing summary file: {e}")
    
    def report_stats(self):
        """Report prioritization statistics"""
        logger.info(f"Prioritization Statistics:")
        logger.info(f"  Total variants processed: {self.stats['total_variants']}")
        logger.info(f"  Pathogenic variants: {self.stats['pathogenic']}")
        logger.info(f"  Likely pathogenic variants: {self.stats['likely_pathogenic']}")
        logger.info(f"  VUS: {self.stats['vus']}")
        logger.info(f"  Likely benign variants: {self.stats['likely_benign']}")
        logger.info(f"  Benign variants: {self.stats['benign']}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Clinical Variant Prioritization for WES Data')
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help='Input TSV file with variants')
    parser.add_argument('-o', '--output', required=True, help='Output TSV file with prioritized variants')
    
    # Optional filtering arguments
    parser.add_argument('-n', '--top', type=int, default=0, help='Number of top variants to include (0 for all)')
    parser.add_argument('-g', '--genes', help='File with list of genes of interest, one per line')
    parser.add_argument('-p', '--phenotype', help='File with HPO terms or phenotype keywords, one per line')
    parser.add_argument('-ih', '--inheritance', help='File with gene-specific inheritance patterns')
    parser.add_argument('-ib', '--include-benign', action='store_true', help='Include benign variants in output')
    parser.add_argument('-c', '--min-cadd', type=float, default=0, help='Minimum CADD phred score (default: 0)')
    parser.add_argument('-af', '--max-gnomad', type=float, default=1.0, 
                        help='Maximum gnomAD global allele frequency (default: 1.0)')
    
    # Output format options
    parser.add_argument('-f', '--format', choices=['tsv', 'excel', 'json'], default='tsv',
                       help='Output format (default: tsv)')
    
    # Logging options
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Configure logging level based on verbose/quiet flags
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    
    return args


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run prioritization
    prioritizer = VariantPrioritization(args)
    success = prioritizer.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
