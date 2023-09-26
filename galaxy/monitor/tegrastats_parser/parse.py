import csv
from http.cookiejar import LWPCookieJar
import os
import re

class Parse:
    def __init__(self, interval, log_file):
        self.interval = int(interval)
        self.log_file = log_file

    def parse_ram(self, lookup_table, ram):
        lookup_table['Used RAM (MB)'] = float(ram[0])
        lookup_table['Total RAM (MB)'] = float(ram[1])
        lookup_table['Number of Free RAM Blocks'] = float(ram[2])
        lookup_table['Size of Free RAM Blocks (MB)'] = float(ram[3])
        return lookup_table

    def parse_swap(self, lookup_table, swap):
        lookup_table['Used SWAP (MB)'] = float(swap[0])
        lookup_table['Total SWAP (MB)'] = float(swap[1])
        lookup_table['Cached SWAP (MB)'] = float(swap[2])
        return lookup_table

    def parse_iram(self, lookup_table, iram):
        lookup_table['Used IRAM (kB)'] = float(iram[0])
        lookup_table['Total IRAM (kB)'] = float(iram[1])
        lookup_table['Size of IRAM Blocks (kB)'] = float(iram[2])
        return lookup_table

    def parse_cpus(self, lookup_table, cpus):
        frequency = re.findall(r'@([0-9]*)', cpus)
        lookup_table['CPU Frequency (MHz)'] = float(frequency[0]) if frequency else ''
        for i, cpu in enumerate(cpus.split(',')):
            lookup_table[f'CPU {i} Load (%)'] = cpu.split('%')[0]
        return lookup_table

    def parse_gr3d(self, lookup_table, gr3d):
        lookup_table['Used GR3D (%)'] = float(gr3d[0])
        lookup_table['GR3D Frequency (MHz)'] = float(gr3d[1]) if gr3d[1] else ''
        return lookup_table

    def parse_emc(self, lookup_table, emc):
        lookup_table['Used EMC (%)'] = float(emc[0])
        lookup_table['GR3D Frequency (MHz)'] = float(emc[1])  if emc[1] else ''
        return lookup_table

    def parse_temperatures(self, lookup_table, temperatures):
        for label, temperature in temperatures:
            lookup_table[f'{label} Temperature (C)'] = float(temperature)
        return lookup_table

    def parse_vdds(self, lookup_table, vdds):
        for label, curr_vdd, avg_vdd in vdds:
            lookup_table[f'Current {label} Power Consumption (mW)'] = float(curr_vdd)
            lookup_table[f'Average {label} Power Consumption (mW)'] = float(avg_vdd)
        return lookup_table
    
    def parse_poms(self, lookup_table, poms):
        for label, curr_pom, avg_pom in poms:
            lookup_table[f'Current {label}'] = float(curr_pom)
            lookup_table[f'Average {label}'] = float(avg_pom)
        return lookup_table

    def parse_data(self, line):
        """ parse tegrastats line to lookup_table"""
        lookup_table = {}

        ram = re.findall(r'RAM ([0-9]*)\/([0-9]*)MB \(lfb ([0-9]*)x([0-9]*)MB\)', line)
        self.parse_ram(lookup_table, ram[0]) if ram else None

        swap = re.findall(r'SWAP ([0-9]*)\/([0-9]*)MB \(cached ([0-9]*)MB\)', line)
        self.parse_swap(lookup_table, swap[0]) if swap else None

        iram = re.findall(r'IRAM ([0-9]*)\/([0-9]*)kB \(lfb ([0-9]*)kB\)', line)
        self.parse_iram(lookup_table, iram[0]) if iram else None

        cpus = re.findall(r'CPU \[(.*)\]', line)
        self.parse_cpus(lookup_table, cpus[0]) if cpus else None

        ape = re.findall(r'APE ([0-9]*)', line)
        if ape:
            lookup_table['APE frequency (MHz)'] = float(ape[0])

        gr3d = re.findall(r'GR3D_FREQ ([0-9]*)%@?([0-9]*)?', line)
        self.parse_gr3d(lookup_table, gr3d[0]) if gr3d else None

        emc = re.findall(r'EMC_FREQ ([0-9]*)%@?([0-9]*)?', line)
        self.parse_emc(lookup_table, emc[0]) if emc else None

        nvenc = re.findall(r'NVENC ([0-9]*)', line)
        if nvenc:
            lookup_table['NVENC frequency (MHz)'] = float(nvenc[0])

        mts = re.findall(r'MTS fg ([0-9]*)% bg ([0-9]*)%', line) # !!!!

        temperatures = re.findall(r'([A-Za-z]*)@([0-9.]*)C', line)
        self.parse_temperatures(lookup_table, temperatures)
        
        poms = re.findall(r'(POM_[A-Za-z0-9_]*_[A-Za-z0-9_]*) ([0-9]*)\/([0-9]*)', line)
        self.parse_poms(lookup_table, poms)

        return lookup_table
