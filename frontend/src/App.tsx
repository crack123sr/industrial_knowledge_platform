import { useState, useRef, useEffect, type FormEvent } from 'react';
import axios from 'axios';
import { copilotAPI } from './apiService';
import { Bot, User, Search, Loader2, Upload, FileText, CheckCircle2, ShieldAlert, BookOpen, Wrench, ArrowRight } from 'lucide-react';

type ChatMessage =
  | { role: 'user'; content: string }
  | {
      role: 'agent';
      responseObj: {
        query_type?: string;
        summary: string;
        extracted_text_chunks?: string[];
        knowledge_graph_relations?: string[];
        recommended_actions?: string[];
      };
      sources?: string[];
      kgUsed?: boolean;
    };

type UploadStatus = { success: boolean; message: string } | null;

const getErrorMessage = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;

    if (typeof detail === 'string') {
      if (detail.toLowerCase().includes('unavailable') || detail.toLowerCase().includes('503')) {
        return 'The AI service is temporarily unavailable. Please try again in a moment.';
      }
      return detail;
    }

    if (error.code === 'ERR_NETWORK') {
      return 'Unable to reach the backend. Make sure the FastAPI server is running.';
    }
  }

  return 'Sorry, I could not process that request right now.';
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'upload'>('chat');
  const [query, setQuery] = useState('');
  const [equipmentId, setEquipmentId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // File Upload States
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSearch = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setQuery('');
    setIsLoading(true);

    try {
      const data = await copilotAPI.query(userMessage, equipmentId || undefined);

      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          responseObj: data.response,
          sources: data.sources_used,
          kgUsed: data.kg_data_used,
        },
      ]);
    } catch (error) {
      console.error('API Error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          responseObj: {
            query_type: 'Error',
            summary: getErrorMessage(error),
            extracted_text_chunks: [],
            knowledge_graph_relations: [],
            recommended_actions: [],
          },
          sources: [],
          kgUsed: false,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const response = await copilotAPI.uploadDocument(selectedFile);
      setUploadStatus({
        success: true,
        message: 'File uploaded successfully',
      });
      setSelectedFile(null);
    } catch (error: unknown) {
      console.error('Upload Error:', error);
      const message = getErrorMessage(error);

      setUploadStatus({
        success: false,
        message,
      });
    } finally {
      setIsUploading(false);
    }
  };
  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Top Header Navigation */}
      <header className="w-full border-b border-slate-800 bg-slate-950/50 backdrop-blur px-8 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/30">
            IK
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">Industrial Knowledge Platform</h1>
            <p className="text-xs text-slate-400">GraphRAG AI Copilot for Technical Manuals</p>
          </div>
        </div>

        <div className="flex gap-2 bg-slate-900 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${activeTab === 'chat' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
          >
            Copilot Chat
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${activeTab === 'upload' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
          >
            Upload Manuals
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 flex flex-col">
        {activeTab === 'chat' ? (
          <div className="flex-1 bg-slate-950 rounded-2xl border border-slate-800 shadow-xl overflow-hidden flex flex-col h-[78vh]">
            {/* Chat Message Scroll Window */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center text-slate-500 space-y-3">
                  <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-blue-500 shadow-inner">
                    <BookOpen size={32} />
                  </div>
                  <h3 className="text-slate-300 font-medium">Ready to query your industrial library</h3>
                  <p className="text-xs max-w-sm">Ask about safety workflows, maintenance routines, or specify an asset tag to inspect component graphs.</p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'agent' && (
                      <div className="w-8 h-8 rounded-xl bg-blue-600 flex items-center justify-center text-white shrink-0 shadow-md">
                        <Bot size={18} />
                      </div>
                    )}

                    <div className={`rounded-2xl max-w-[85%] p-5 shadow-md ${
                      msg.role === 'user' 
                        ? 'bg-blue-600 text-white rounded-tr-none' 
                        : 'bg-slate-900 border border-slate-800 rounded-tl-none text-slate-200'
                    }`}>
                      {msg.role === 'user' ? (
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      ) : msg.responseObj ? (
                        <div className="space-y-4">
                          {/* Query Category Badge */}
                          <div className="flex items-center justify-between">
                            <span className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold rounded-full uppercase tracking-wider">
                              {msg.responseObj.query_type || "General Inquiry"}
                            </span>
                            {msg.kgUsed && (
                              <span className="text-xs text-emerald-400 font-medium flex items-center gap-1">
                                <CheckCircle2 size={12} /> Knowledge Graph Linked
                              </span>
                            )}
                          </div>

                          {/* Summary Text */}
                          <p className="text-sm text-slate-100 font-medium leading-relaxed">
                            {msg.responseObj.summary}
                          </p>

                          {/* Extracted Document Chunks */}
                          {msg.responseObj.extracted_text_chunks && msg.responseObj.extracted_text_chunks.length > 0 && (
                            <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80 space-y-2">
                              <p className="text-xs font-bold text-slate-400 flex items-center gap-1.5 uppercase tracking-wider">
                                <FileText size={14} className="text-blue-400"/> Manual Reference Excerpts:
                              </p>
                              <ul className="text-xs space-y-1.5 text-slate-300 list-disc list-inside">
                                {msg.responseObj.extracted_text_chunks.map((chunk, i) => (
                                  <li key={i} className="leading-normal">{chunk}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Knowledge Graph Relations */}
                          {msg.responseObj.knowledge_graph_relations && msg.responseObj.knowledge_graph_relations.length > 0 && (
                            <div className="bg-emerald-950/20 p-3.5 rounded-xl border border-emerald-900/30 space-y-2">
                              <p className="text-xs font-bold text-emerald-400 flex items-center gap-1.5 uppercase tracking-wider">
                                <Wrench size={14}/> Graph Relations:
                              </p>
                              <ul className="text-xs space-y-1 text-emerald-300/90 font-mono">
                                {msg.responseObj.knowledge_graph_relations.map((rel, i) => (
                                  <li key={i}>• {rel}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Recommended Actions */}
                          {msg.responseObj.recommended_actions && msg.responseObj.recommended_actions.length > 0 && (
                            <div className="bg-amber-950/20 p-3.5 rounded-xl border border-amber-900/30 space-y-2">
                              <p className="text-xs font-bold text-amber-400 flex items-center gap-1.5 uppercase tracking-wider">
                                <ShieldAlert size={14}/> Recommended Operator Actions:
                              </p>
                              <ul className="text-xs space-y-1 text-amber-200/90 list-disc list-inside">
                                {msg.responseObj.recommended_actions.map((act, i) => (
                                  <li key={i}>{act}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Sources Cited */}
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="pt-3 border-t border-slate-800/80 flex items-center gap-2 text-xs text-slate-500">
                              <span className="font-medium">Sources:</span>
                              {msg.sources.map((src, i) => (
                                <span key={i} className="px-2 py-0.5 bg-slate-800 rounded text-slate-400 font-mono">{src}</span>
                              ))}
                            </div>
                          )}
                        </div>
                      ) : null}
                    </div>

                    {msg.role === 'user' && (
                      <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 shadow-md">
                        <User size={18} />
                      </div>
                    )}
                  </div>
                ))
              )}

              {isLoading && (
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-xl bg-blue-600 flex items-center justify-center text-white shrink-0">
                    <Loader2 size={18} className="animate-spin" />
                  </div>
                  <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 text-slate-400 text-sm italic flex items-center gap-3">
                    <Loader2 size={16} className="animate-spin text-blue-500" /> Traversing vector embeddings & knowledge graph...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input Toolbar */}
            <div className="p-4 bg-slate-900 border-t border-slate-800">
              <form onSubmit={handleSearch} className="flex gap-3">
                <input
                  type="text"
                  placeholder="Optional Asset ID (e.g. Spindle)"
                  value={equipmentId}
                  onChange={(e) => setEquipmentId(e.target.value)}
                  className="w-1/4 px-4 py-3 bg-slate-950 border border-slate-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-slate-200 placeholder-slate-600"
                />
                <div className="relative flex-1">
                  <input
                    type="text"
                    placeholder="Ask a technical or safety question about your manual..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-950 border border-slate-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 pl-11 text-sm text-slate-200 placeholder-slate-600 shadow-inner"
                    disabled={isLoading}
                  />
                  <Search className="absolute left-4 top-3.5 text-slate-500" size={18} />
                </div>
                <button
                  type="submit"
                  disabled={isLoading || !query.trim()}
                  className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-500 disabled:opacity-50 transition-all shadow-lg shadow-blue-600/20 flex items-center gap-2 text-sm"
                >
                  <span>Ask Copilot</span>
                  <ArrowRight size={16} />
                </button>
              </form>
            </div>
          </div>
        ) : (
          /* Document Upload Workspace */
          <div className="flex-1 bg-slate-950 rounded-2xl border border-slate-800 shadow-xl p-8 flex flex-col items-center justify-center">
            <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl flex flex-col items-center text-center">
              <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-4">
                <Upload size={28} />
              </div>
              <h2 className="text-xl font-bold text-white mb-2">Upload Industrial Manual</h2>
              <p className="text-xs text-slate-400 mb-6">Drop any general machine PDF manual to ingest it into the RAG pipeline instantly.</p>

              <form onSubmit={handleFileUpload} className="w-full space-y-4">
                <label className="border-2 border-dashed border-slate-700 hover:border-blue-500 transition-all rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer bg-slate-950/50">
                  <FileText className="text-slate-500 mb-2" size={32} />
                  <span className="text-xs text-slate-300 font-medium">
                    {selectedFile ? selectedFile.name : "Click to browse PDF file"}
                  </span>
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    className="hidden"
                  />
                </label>

                <button
                  type="submit"
                  disabled={!selectedFile || isUploading}
                  className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-semibold rounded-xl transition-all shadow-lg shadow-blue-600/20 flex items-center justify-center gap-2 text-sm"
                >
                  {isUploading ? (
                    <>
                      <Loader2 size={16} className="animate-spin" /> Uploading PDF...
                    </>
                  ) : (
                    "Upload and Save PDF"
                  )}
                </button>
              </form>

              {uploadStatus && (
                <div className={`mt-4 p-3 rounded-xl text-xs font-medium border ${
                  uploadStatus.success 
                    ? 'bg-emerald-950/30 border-emerald-900/50 text-emerald-400' 
                    : 'bg-rose-950/30 border-rose-900/50 text-rose-400'
                }`}>
                  {uploadStatus.message}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}