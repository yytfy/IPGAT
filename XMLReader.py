import xml.etree.ElementTree
import time

class drug_xml_reader():

    xml_file_name = 'full database.xml'

    def __init__(self,file_name,zipped=True):
        self.file_name=file_name
        self.zipped=zipped
        if zipped==False:
            assert False, 'didnt implement yet unzipped handling' #if i will need it, just read to file instead of unzipping it

        self.all_drugs = []
        self.drug_to_interactions = {}
        self.drug_id_to_name = {}

    def read_data_from_file(self):
        print('reading file:' + self.file_name)
        start_time = time.time()

        if self.zipped:
            import zipfile
            archive = zipfile.ZipFile(self.file_name, 'r')
            db_file = archive.open(self.xml_file_name)

        root = xml.etree.ElementTree.parse(db_file).getroot()
        elapsed_time = time.time() - start_time
        ns = '{http://www.drugbank.ca}'
        for i, drug in enumerate(root):
            genname = []
            assert drug.tag == '{ns}drug'.format(ns=ns)
            drug_p_id = drug.findtext("{ns}drugbank-id[@primary='true']".format(ns=ns))
            assert drug_p_id not in self.all_drugs
            assert len(drug.findall("{ns}groups".format(ns=ns))) == 1
            drug_approved=False
            for j,group in enumerate(drug.findall("{ns}groups".format(ns=ns))[0].getchildren()):
                if 'approved' == group.text:
                    drug_approved=True
            if drug_approved:
                self.all_drugs.append(drug_p_id)
                self.drug_id_to_name[drug_p_id] = drug.findtext("{ns}name".format(ns=ns))
                assert len(drug.findall("{ns}drug-interactions".format(ns=ns))) == 1
                for interaction in drug.findall("{ns}drug-interactions".format(ns=ns))[0].getchildren():
                    self.drug_to_interactions.setdefault(drug_p_id, set()).add(interaction.findtext('{ns}drugbank-id'.format(ns=ns)))

        print('time to read file:', elapsed_time)
        print('number of drugs read:', len(self.all_drugs))
        print('number of drugs with interactions', len(self.drug_to_interactions))
        print('drugs with no interactions:', len(set(self.all_drugs) - set(self.drug_to_interactions.keys())))
        print('Finish xml reading----------------------------------------------------------------------')
