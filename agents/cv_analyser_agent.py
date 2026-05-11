from agents.rag_agent import DataCard, RagAgent

from langchain_core.messages import HumanMessage, SystemMessage


class CvAnalyserAgent(RagAgent):
    
    def index_cv(self, path: str, data_card: DataCard) -> None:
        """Index a CV document into the document store with the provided data card metadata.
        Args:
            path (str): The file path to the CV document to be indexed.
            data_card (DataCard): Metadata about the document's provenance and sensitivity.
        """
        # Attach the provided data card metadata to the document
        metadata = {
            "source": data_card.source,
            "license": data_card.license,
            "pii_risk": data_card.pii_risk,
            "refresh_cadence": data_card.refresh_cadence,
        }
        
        # Load the CV document from the specified path, applying PII redaction
        docs = self._store.load_file(path, metadata=metadata)

        # Index the CV content into the document store with metadata
        self._store.index(documents=docs)
        
    
    def summarise_cv(self) -> str:
        """ Produce a structured summary of the candidate on demand """   
        return self._rag.answer(
            "Provide a structured summary of this candidate with exactly these three fields:\n"
            "Name: <candidate full name>\n"
            "Experience Level: <junior / mid / senior>\n"
            "Top 3 Skills: <comma separated list>"
        )
        
    
    def extract_skills(self) -> str:
        """ Extract technical and soft skills as separate lists """
        return self._rag.answer(                                                                                                                                                                                  
            "List this candidate's skills in two separate sections.\n"                                                                                                                                            
            "Use exactly these headers:\n"
            "TECHNICAL SKILLS:\n"
            "- list each technical skill as a bullet point\n\n"
            "SOFT SKILLS:\n"
            "- list each soft skill as a bullet point"
        )
    
    
    def suggest_improvements(self, section: str) -> str:
        """ Suggest improvements for a named section (e.g. SUMMARY, WORK EXPERIENCE) """
        return self._rag.answer(                                                                                                                                                                                  
              f"Review the {section} section of this candidate's CV.\n"                                                                                                                                             
              "Suggest exactly three specific improvements with a reason for each.\n"
              "Format your response as:\n"
              "1. <improvement> — <reason>\n"
              "2. <improvement> — <reason>\n"
              "3. <improvement> — <reason>"
        )
    
    def gap_analysis(self, job_description_path: str) -> str:    
        """ Gap analysis capability — given a job description, identify the candidate's top three skill or experience gaps.
        Compare CV against a job description and return the top 3 gaps.""" 
                                                                                                                                                                       
        jd_text = open(job_description_path, encoding="utf-8").read()                                                                                                                                                          
                                                                                                                                                                                                                
        cv_chunks = self._store.retrieve("skills experience qualifications", top_k=5)
        cv_context = "\n\n".join(r.document.page_content for r in cv_chunks)

        messages = [
            SystemMessage(content="You are a hiring assistant. Be specific and concise."),
            HumanMessage(content=(
                f"Job Description:\n{jd_text}\n\n"
                f"Candidate CV (relevant excerpts):\n{cv_context}\n\n"
                "Identify the top 3 skill or experience gaps between this candidate "
                "and the job description. Format as:\n"
                "1. <gap> — <why it matters>\n"
                "2. <gap> — <why it matters>\n"
                "3. <gap> — <why it matters>"
            ))
        ]
        return self._llm.invoke(messages)